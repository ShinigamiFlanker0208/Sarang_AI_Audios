import jax

import jax.numpy as jnp

import tensorflow as tf

from jax.experimental import jax2tf

from pitch_kernal import compute_difference_function, compute_cmnd, extract_pitch





# 1. Define the Full Pipeline

def sarang_pitch_pipeline(audio_frame):

    diffs = compute_difference_function(audio_frame)

    cmnd = compute_cmnd(diffs)

    # We use the interpolated version to get that sub-sample accuracy!

    pitch_hz = extract_pitch(cmnd, 44100.0, threshold=0.10)

    return pitch_hz





if __name__ == "__main__":

    print("🚀 Starting DeepMind TFLite Export Protocol...")



    # 2. Convert JAX graph to TensorFlow

    # enable_xla=True is key for your 40-series card performance

    tf_fn = jax2tf.convert(sarang_pitch_pipeline, enable_xla=True)





    # 3. Create a wrapper with a FIXED input signature (2048)

    @tf.function(input_signature=[tf.TensorSpec(shape=[2048], dtype=tf.float32)])

    def serving_fn(audio_frame):

        return tf_fn(audio_frame)





    # 4. Convert to TFLite with SELECT_TF_OPS

    # This ensures all JAX/XLA math is supported in the mobile runtime

    print("Optimizing for Edge (TFLite)...")

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])



    # This allows the model to use complex math ops if needed

    converter.target_spec.supported_ops = [

        tf.lite.OpsSet.TFLITE_BUILTINS,

        tf.lite.OpsSet.SELECT_TF_OPS

    ]



    tflite_model = converter.convert()



    # 5. Save to Disk

    file_name = "sarang_pitch_model.tflite"

    with open(file_name, "wb") as f:

        f.write(tflite_model)



    print(f"✅ Success! Brain frozen and saved as: {file_name}")