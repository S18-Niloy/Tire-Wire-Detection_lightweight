import os, glob, cv2, numpy as np
import tensorflow as tf

tflite = tf.lite

# Paths
FRAMES_ROOT = "output/frames_2fps"
TFLITE_MODEL = "exports/model_int8.tflite"
OUTPUT_ROOT = "classified_frames"  # where to save frames
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABELS = [l.strip() for l in open("exports/labels.txt", "r", encoding="utf-8").read().splitlines()]

# TFLite setup
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
inp_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()
H = W = int(inp_details[0]['shape'][1])

def classify(img_bgr):
    img = cv2.resize(img_bgr, (W, H))
    inp = img.astype(np.uint8)[None, ...]
    interpreter.set_tensor(inp_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(out_details[0]['index'])[0]
    if out.dtype == np.uint8:
        probs = out.astype(np.float32) / 255.0
    else:
        probs = out.astype(np.float32)
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx])

# Traverse nested folders
for video_dir in glob.glob(os.path.join(FRAMES_ROOT, "**"), recursive=True):
    if not os.path.isdir(video_dir): 
        continue
    print("Classifying crops in:", video_dir)

    for img_path in glob.glob(os.path.join(video_dir, "*.jpg")):
        det_txt = img_path + ".txt"
        img = cv2.imread(img_path)
        if img is None:
            continue

        if not os.path.exists(det_txt):
            # No detection file; classify whole frame
            cls, conf = classify(img)
            print(os.path.basename(img_path), "FULLFRAME:", cls, conf)
            
            # Save the frame in folder named by predicted label
            save_dir = os.path.join(OUTPUT_ROOT, cls)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)
            continue

        with open(det_txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for i, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 6: 
                continue
            cls_name = parts[0]
            x1, y1, x2, y2 = map(int, map(float, parts[1:5]))
            score = float(parts[5])

            crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if crop.size == 0:
                continue
            pred, pconf = classify(crop)
            print(os.path.basename(img_path), f"DET{i}:{cls_name}@{score:.2f} -> CLASS:{pred} ({pconf:.2f})")
            
            # Save the crop in folder named by predicted label
            save_dir = os.path.join(OUTPUT_ROOT, pred)
            os.makedirs(save_dir, exist_ok=True)
            crop_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_DET{i}.jpg"
            save_path = os.path.join(save_dir, crop_name)
            cv2.imwrite(save_path, crop)
