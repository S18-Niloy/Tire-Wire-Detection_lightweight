import tensorflow as tf
import random
from typing import List


def build_model(num_classes: int, img_size: int, base_trainable: bool) -> tuple[tf.keras.Model, tf.keras.Model]:

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), dtype="uint8", name="images_uint8")

    x = tf.keras.layers.Rescaling(scale=1/127.5, offset=-1.0, name="to_minus1_1")(tf.cast(inputs, tf.float32))
    
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg"
    )
    base.trainable = base_trainable
    
    x = base.output
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model, base



def representative_data_gen(paths: List[str], img_size: int, n_samples: int = 200):

    rng = random.Random(123)
    samples = paths.copy()
    rng.shuffle(samples)
    samples = samples[:max(1, min(n_samples, len(samples)))]
    
    for p in samples:
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        img = tf.expand_dims(img, 0)
        yield [img.numpy()]  # uint8 input for full-integer quantization



def export_tflite_models(saved_model_dir: str, export_dir: str, train_paths: List[str], img_size: int):

    import os

    # FP32 TFLite
    print("\nExporting FP32 TFLite ...")
    converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_fp32.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_fp32 = converter_fp32.convert()
    out_fp32 = os.path.join(export_dir, "model_fp32.tflite")
    with open(out_fp32, "wb") as f:
        f.write(tflite_fp32)
    print("Saved:", out_fp32)

    # INT8 quantized TFLite
    print("Exporting INT8 TFLite ...")
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = lambda: representative_data_gen(train_paths, img_size=img_size, n_samples=200)
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.uint8
    converter_int8.inference_output_type = tf.uint8
    tflite_int8 = converter_int8.convert()
    out_int8 = os.path.join(export_dir, "model_int8.tflite")
    with open(out_int8, "wb") as f:
        f.write(tflite_int8)
    print("Saved:", out_int8)

    print("\nTFLite Export Complete!")
    print(" -", out_fp32)
    print(" -", out_int8, "  <-- deploy this on Raspberry Pi 3")
