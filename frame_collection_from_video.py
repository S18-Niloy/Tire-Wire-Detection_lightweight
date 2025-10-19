import cv2, os, glob

# Root folder (Tire Data Video Collected)
IN_DIR = r"C:\Users\T2420350\Downloads\P3 Final Data with  Break in Update\P3 Final Data with  Break in Update\Tire Data Video Collected"
OUT_ROOT = "output/frames_2fps"

os.makedirs(OUT_ROOT, exist_ok=True)

# Find all .MOV files in all subfolders
video_files = glob.glob(os.path.join(IN_DIR, "**", "*.MOV"), recursive=True)

for path in video_files:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open:", path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // 2)  # 2 frames per second
    if frame_interval < 1:
        frame_interval = 1

    # Preserve subfolder structure in output
    rel_path = os.path.relpath(path, IN_DIR)
    rel_dir = os.path.dirname(rel_path)
    bname = os.path.splitext(os.path.basename(path))[0]

    outdir = os.path.join(OUT_ROOT, rel_dir, bname)
    os.makedirs(outdir, exist_ok=True)

    idx = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            idx += 1
            save_path = os.path.join(outdir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(save_path, frame)
        frame_idx += 1
    cap.release()
    print(f"{path} -> saved {idx} frames into {outdir}")
