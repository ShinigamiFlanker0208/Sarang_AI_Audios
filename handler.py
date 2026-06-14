import runpod
import base64
import os
import subprocess
import librosa
import numpy as np
import soundfile as sf
import math
import torch

UPLOAD_DIR = "temp_uploads"
OUTPUT_DIR = "temp_separated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handler(job):
    job_input = job["input"]

    # Decode the base64 audio file sent from Flutter
    audio_bytes = base64.b64decode(job_input["file_b64"])
    filename = job_input["filename"]          # e.g. "mysong.mp3"
    user_max_hz = float(job_input.get("user_max_hz", 330.0))

    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    # Run Demucs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subprocess.run([
        "demucs", "--two-stems=vocals",
        "-n", "htdemucs",
        "-d", device,
        "-o", OUTPUT_DIR,
        file_path
    ], check=True)

    song_name = os.path.splitext(filename)[0]
    vocal_path = os.path.join(OUTPUT_DIR, "htdemucs", song_name, "vocals.wav")
    instrumental_path = os.path.join(OUTPUT_DIR, "htdemucs", song_name, "no_vocals.wav")

    # Extract pitch
    y, sr = librosa.load(vocal_path, sr=44100)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=50, fmax=1500, sr=sr, frame_length=2048
    )

    ghost_line = []
    for i, freq in enumerate(f0):
        ghost_line.append({
            "frame": i,
            "time_sec": round(i * (2048 / sr), 3),
            "hz": round(float(freq), 2) if (voiced_flag[i] and not np.isnan(freq)) else 0.0
        })

    # Auto key shift
    song_max_hz = max([p["hz"] for p in ghost_line if 50.0 < p["hz"] < 950.0], default=0)
    semitone_shift = 0
    if song_max_hz > 0 and user_max_hz > 0:
        raw_shift = 12.0 * math.log2(user_max_hz / song_max_hz)
        semitone_shift = (round(raw_shift) + 6) % 12 - 6

    if semitone_shift != 0:
        shift_multiplier = 2.0 ** (semitone_shift / 12.0)
        for point in ghost_line:
            if point["hz"] > 0:
                point["hz"] = round(point["hz"] * shift_multiplier, 2)
        y_inst, sr_inst = librosa.load(instrumental_path, sr=44100)
        y_shifted = librosa.effects.pitch_shift(y=y_inst, sr=sr_inst, n_steps=semitone_shift)
        sf.write(instrumental_path, y_shifted, sr_inst)

    # Encode instrumental as base64 to send back
    with open(instrumental_path, "rb") as f:
        instrumental_b64 = base64.b64encode(f.read()).decode("utf-8")

    os.remove(file_path)

    return {
        "status": "success",
        "instrumental_b64": instrumental_b64,
        "ghost_line": ghost_line
    }

runpod.serverless.start({"handler": handler})