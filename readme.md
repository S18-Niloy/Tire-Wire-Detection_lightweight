# 🎬 Custom Video Processing Model

This repository contains code for a **custom video processing and analysis pipeline**, including dataset preprocessing, frame extraction from videos, and Grad-CAM visualization for model interpretability. The dataset used is **proprietary and created by the author**, so it is not included in this repository.

---

## 📦 Project Structure

```
CustomVideoModel/
│
├── dataset_preprocessing.py      # Script to preprocess your dataset
├── frame_collection_from_video.py # Extract frames from input videos
├── gradCAM.pmodel.py             # Grad-CAM model implementation
├── testing_on_video.py           # Script to test model on video
├── setup.py                      # Project setup and dependencies
└── README.md                     # Project documentation
```

---

## 🧩 Features

- Video frame extraction
- Custom dataset preprocessing
- Grad-CAM visualization for model interpretability
- Testing pipeline for video input
- Comparative model analysis for benchmarking

---

## 📦 Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or if using `setup.py`:

```bash
python setup.py install
```

---

## 🧠 Usage

### Dataset Preprocessing

```bash
python dataset_preprocessing.py --input path/to/raw/data --output path/to/preprocessed/data
```

### Frame Extraction from Video

```bash
python frame_collection_from_video.py --video path/to/video.mp4 --output path/to/frames
```

### Grad-CAM Visualization

```bash
python gradCAM.pmodel.py --model path/to/model.pth --input path/to/image_or_frame
```

### Testing on Video

```bash
python testing_on_video.py --video path/to/video.mp4 --model path/to/model.pth --output path/to/results
```

---

## 📊 Comparative Model Analysis

| Model                           | Train Acc (%) | Val Acc (%) | Latency (ms) | GFLOPs |
|---------------------------------|---------------|-------------|--------------|--------|
| T2T-ViT-14                      | 89.66         | 80.29       | 49.25        | 4.5    |
| CSWin-T                          | 77.55         | 75.87       | 71.67        | 5.5    |
| MaxSA-T                          | 72.33         | 68.44       | 56           | 3.0    |
| GFNet-Ti                         | 94.33         | 89.44       | 79           | 5.8    |
| FocalNet-Tiny                    | 90.34         | 82.18       | 67           | 4.8    |
| SqueezeNet                        | 82.56         | 71.37       | 53           | 1.2    |
| **EfficientLiteNetB0 (Proposed)** | **99.67**    | **94.24**  | **45**       | **0.7** |

This table provides a **benchmark comparison** of the proposed model against several Transformer-based and CNN models in terms of training accuracy, validation accuracy, latency, and GFLOPs.

---

## ⚙️ Notes

- The dataset is proprietary and not included in this repository.
- Ensure all videos are in supported formats (e.g., `.mp4`, `.avi`).
- Adjust model paths and preprocessing parameters as required.

---

## 🧑‍💻 Contributors

| Name                     | Role                |
| ------------------------ | ------------------ |
| **Your Name**            | Author / Developer |

---

## 🪪 License

This project is licensed under the MIT License.

---

> *Custom Video Processing Model — Developed for proprietary dataset and research purposes.*

