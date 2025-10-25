---
title: "LBPH Face Recognition with OpenCV"
date: 2025-10-25
categories: [Projects, Computer Vision, AI]
tags: [Projects, OpenCV, Face Recognition, LBPH, Python]
---

# LBPH Face Recognition with OpenCV — An approachable end-to-end article

Face recognition is an accessible entry point to computer vision: it combines classic image processing, dataset curation, and straightforward model training. This project demonstrates a compact, practical pipeline built with OpenCV's LBPH (Local Binary Patterns Histograms) recognizer. This article explains the motivation, the ideas behind each stage, and how the pieces connect, so you can reproduce it or use it as a learning resource.

Check the project's repo on GitHub : https://github.com/ammarlouah/face-recognition-opencv-lbph

---

## Why LBPH? A pragmatic pick for small-scale face recognition

Modern deep-learning models dominate face recognition benchmarks, but they come with heavy compute requirements and long training times. LBPH offers a lightweight, interpretable alternative:

- **Fast training and inference** : works comfortably on CPUs and small devices.
- **Robust to monotonic illumination changes** because of local binary patterns.
- **Easy to implement with OpenCV** : great for demos, embedded projects, and learning.

This project is intentionally a focused demo: capture faces from a webcam, train an LBPH model, evaluate it and run live recognition, all with vanilla Python and OpenCV.

---

## Project highlights (what you get)

- A capture tool that records face crops and metadata while checking for stable detections.
- A trainer that preprocesses faces (resize + histogram equalization), encodes labels, and trains an LBPH model; saves both model and label map.
- An evaluator that loads a held-out test split and prints a classification report and confusion matrix.
- A live recognition script that runs a webcam demo, shows names + confidence values, and can save unknown faces.

---

## How it works : the pipeline, step by step

### 1. Data capture (`store_data.py`)

Good data is half the work.

- The capture script opens your webcam, detects faces using a Haar cascade and picks the largest detection (assumed primary subject).
- To avoid noisy samples, the script computes a **bbox similarity** over a short window and only saves frames when the face is stable (similar box across several frames).
- Each saved sample is also optionally augmented (horizontal flip, slight brightness changes, small rotations) to boost robustness without complicated augmentation libraries.
- Saved assets: cropped, resized images under `dataset/` and a `dataset/metadata.csv` with label info.

Why stability checks? They reduce motion blur and partial detections, improving model quality even with a small number of samples.

### 2. Training (`train_lbph.py`)

- The trainer loads dataset images and normalizes them with `cv2.equalizeHist` (histogram equalization on grayscale) : this helps with lighting variance.
- Labels are encoded (e.g., `user01` → integer id). The script supports datasets organized either as per-label subfolders or flat filenames like `User.<label>.<n>.jpg`.
- The LBPH recognizer parameters are configurable: `radius`, `neighbors`, `grid_x`, `grid_y`. These control the LBP sampling and the grid partitioning used to compute histograms.
- The trainer optionally creates a train/test split (e.g., 80/20) and saves the model (`.yml`) and label map (`.pkl`). It also persists `X_test.npy` and `y_test.npy` if a split is used.

LBPH is not probabilistic. OpenCV returns a **confidence (distance-like)** value where **lower means better match**. That affects how you set thresholds for recognition.

### 3. Evaluation (`evaluate_lbph.py`)

- Using the saved test split, the evaluator predicts labels for each test sample and produces:
  - overall accuracy
  - per-class precision/recall/F1 (classification report)
  - confusion matrix visualization (helps find which identities are frequently confused)

These metrics provide practical feedback: if a subject gets poor recall, consider collecting more varied samples for them.

### 4. Live recognition (`recognize_live.py`)

- The live demo detects faces, applies the same preprocessing (resize + equalize), queries the LBPH model, and displays the predicted name along with the `confidence` score.
- Controls:
  - `q` to quit the demo.
  - `u` to save the currently detected face as `unknown_<timestamp>.jpg` for later inspection.
- Because LBPH confidence is a distance, the script considers a match valid when `confidence < threshold` (default example: `70`). Tune this threshold on your validation split.

---

## Practical tips & best practices

- **Collect varied samples:** different poses, eyewear/no eyewear, small illumination changes and slight rotations. LBPH handles limited illumination variations, but variety still helps.
- **Balanced dataset:** LBPH can be biased if some identities have many more images. Try to keep sample counts comparable.
- **Preprocessing consistency:** the same resizing and histogram equalization must be used during both training and inference.
- **Tune LBPH params:** `grid_x/grid_y` control spatial resolution of histograms. More grids capture finer local patterns but require more data.
- **Threshold tuning:** pick `confidence-threshold` based on the model’s ROC or validation confusion matrix, there’s no universal default.

---

## Troubleshooting & common issues

- **`AttributeError: cv2 has no attribute 'face'`** — install `opencv-contrib-python` (contains `cv2.face`).
- **Camera not accessible** — try different `--cam` index (0,1,2…) or check OS camera permissions.
- **Haarcascade not found** — specify the cascade path via `--cascade` or ensure `haarcascade/haarcascade_frontalface_default.xml` exists.
- **Saving model raises `FileNotFoundError`** — provide output paths that include directories (e.g., `model/lbph_model.yml`) or create the directories beforehand.

---

## Quick commands (concise)

```bash
# Clone the repository from GitHub
git clone https://github.com/ammarlouah/face-recognition-opencv-lbph.git
cd face-recognition-opencv-lbph

# Capture samples
python ./store_data.py

# Train model
python ./train_lbph.py --dataset dataset --img-size 200 --test-size 0.2 --model-out model/lbph_model.yml --labels-out model/label_map.pkl

# Evaluate
python ./evaluate_lbph.py --model model/lbph_model.yml --labels model/label_map.pkl --x-test model/X_test.npy --y-test model/y_test.npy

# Live recognition
python ./recognize_live.py --model model/lbph_model.yml --labels model/label_map.pkl --confidence-threshold 70
```

---

## Where to go next (extensions)

- Replace LBPH with a lightweight CNN (MobileNet/MiniFace) for improved accuracy on larger datasets.
- Deploy the recognizer to an edge device. LBPH's CPU efficiency makes it a good fit for Raspberry Pi class hardware.
- Add face alignment (eye/landmark localization) before cropping to reduce pose-related errors.

---

## Conclusion

This LBPH demo is designed to be both educational and practical: it teaches core ideas (dataset collection, preprocessing, parameter tuning) and provides working scripts for capture, training, evaluation, and live recognition. Because it relies on OpenCV primitives, it’s easy to inspect, modify, and extend.

If you reuse or adapt this work in a project, consider improving data diversity and tuning LBPH parameters to your deployment conditions.

---

## Contact

Questions, feedback, or suggestions? Email: ammarlouah9@gmail.com
