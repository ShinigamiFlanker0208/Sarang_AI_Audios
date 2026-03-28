import os
import subprocess
import librosa
import numpy as np
import soundfile as sf   # pip install soundfile — saves the pitch-shifted instrumental
import math              # log2 for semitone calculation
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="Sarang Audio Engine")
app.mount("/temp_separated", StaticFiles(directory="temp_separated"), name="temp_separated")

# Temporary directories for processing
UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_separated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/process-track")
async def process_track(
    file: UploadFile = File(...),
    user_max_hz: float = Form(330.0)  # User's vocal ceiling in Hz (default E4 = 330Hz)
):
    """
    1. Receives the MP3.
    2. Separates the vocals.
    3. Extracts the Ghost Line (Pitch JSON).
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"✅ Received {file.filename}. Starting separation...")

    # 👉 1. THE SEPARATION ENGINE (Demucs)
    # We use the two-stems model to just get vocals and instrumental (saves time)
    command = [
        "demucs",
        "--two-stems=vocals",
        "-n", "htdemucs",
        "-d", "cuda",
        "-o", OUTPUT_DIR,
        file_path
    ]

    # Run Demucs (This will take a moment depending on your CPU/GPU)
    subprocess.run(command, check=True)

    # Demucs creates a folder structure like: temp_separated/htdemucs/{filename}/vocals.wav
    song_name = os.path.splitext(file.filename)[0]
    vocal_path = os.path.join(OUTPUT_DIR, "htdemucs", song_name, "vocals.wav")
    instrumental_path = os.path.join(OUTPUT_DIR, "htdemucs", song_name, "no_vocals.wav")

    print(f"✅ Separation complete. Extracting Ghost Line...")

    # 👉 2. THE PITCH EXTRACTOR (Librosa pYIN)
    # We load the audio at 44100Hz to match our Dart mobile app
    y, sr = librosa.load(vocal_path, sr=44100)

    # We use a frame_length of 2048 to match the 50ms chunks we used in Dart!
    # fmin=50, fmax=1500 matches our Biological Filter.
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=50,
        fmax=1500,
        sr=sr,
        frame_length=2048
    )

    # Convert the numpy arrays to a clean JSON structure
    ghost_line = []

    for i, freq in enumerate(f0):
        if voiced_flag[i] and not np.isnan(freq):
            ghost_line.append({
                "frame": i,
                "time_sec": round(i * (2048 / sr), 3),
                "hz": round(float(freq), 2)
            })
        else:
            # Silent or unvoiced frame
            ghost_line.append({
                "frame": i,
                "time_sec": round(i * (2048 / sr), 3),
                "hz": 0.0
            })

    # Clean up the original upload to save space
    os.remove(file_path)

    print(f"✅ Ghost Line extracted. {len(ghost_line)} frames mapped.")

    # ── 3. AUTO-KEY SHIFT ENGINE ──────────────────────────────────────────────
    # Find the highest note actually sung in the original track
    song_max_hz = max(
        [p["hz"] for p in ghost_line if 50.0 < p["hz"] < 950.0],
        default=0
    )
    semitone_shift = 0

    if song_max_hz > 0 and user_max_hz > 0:
        # Calculate raw semitone difference: 12 * log2(target / source)
        raw_shift = 12.0 * math.log2(user_max_hz / song_max_hz)

        # Fold into shortest musical path (-6 to +5 semitones) so the
        # instrumental stays sounding natural — no Darth Vader effect
        semitone_shift = round(raw_shift)
        semitone_shift = (semitone_shift + 6) % 12 - 6

    if semitone_shift != 0:
        print(f"🎵 Auto-Key: shifting {semitone_shift:+d} semitones "
              f"(song peak {song_max_hz:.0f}Hz → user ceiling {user_max_hz:.0f}Hz)")

        shift_multiplier = 2.0 ** (semitone_shift / 12.0)

        # A. Shift the JSON Ghost Line mathematically (instant, no audio processing)
        for point in ghost_line:
            if point["hz"] > 0:
                point["hz"] = round(point["hz"] * shift_multiplier, 2)

        # B. Pitch-shift the instrumental audio file to match
        y_inst, sr_inst = librosa.load(instrumental_path, sr=44100)
        y_shifted = librosa.effects.pitch_shift(y=y_inst, sr=sr_inst, n_steps=semitone_shift)
        sf.write(instrumental_path, y_shifted, sr_inst)

        print(f"✅ Auto-Key complete. Instrumental and Ghost Line aligned.")
    else:
        print("✅ Track already in user's range. No key shift needed.")

    # ── 4. RETURN THE PAYLOAD ─────────────────────────────────────────────────
    return JSONResponse(content={
        "status": "success",
        "instrumental_path": instrumental_path,
        "ghost_line": ghost_line
    })