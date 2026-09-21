# MathLens

**MathLens** is an end-to-end computer vision pipeline designed to automatically detect, sequence, and crop mathematical expressions from documents, textbooks, and images with high precision.

The project is built on a modular architecture optimized for high-speed inference and minimal memory footprint on CPU environments.

---

## Key Features

- **Advanced Object Detection:** Powered by Ultralytics YOLO11s.
- **ONNX Runtime & INT8 Quantization:** Substantially boosted CPU inference speeds and reduced memory usage via Graph Optimization and Dynamic INT8 Quantization.
- **Smart Model Fallback System:** Dynamically loads the most optimized available model weight at runtime:

  `best_int8.onnx` ➔ `best.onnx` ➔ `best.pt`

- **Spatial Order Sorting:** Sorts detected bounding boxes based on natural reading order (top-to-bottom, left-to-right) for logical file naming (`image-mathlens-01.png`).
- **Flexible Configuration:** Customizable confidence threshold (`conf_threshold`), bounding box padding, and minimum cropping size.

---

## Model Weights & Releases

Pre-trained model weights (`best.pt`, `best.onnx`, and `best_int8.onnx`) and the latest training checkpoint (`last.pt`) are not tracked directly in this Git repository to maintain a lightweight footprint.

You can download the compiled model artifacts and training checkpoints from the [GitHub Releases Page](https://github.com/GurhanAras/MathLens/releases).

Once downloaded, place the weight files inside the `weights/` directory:

```text
weights/
├── best.pt
├── best.onnx
├── best_int8.onnx
└── last.pt                  
```

---

## Project Architecture

```text
MathLens/
├── dataset/                  # Dataset configuration
│   ├── data.yaml             # YOLO dataset labels and path definitions
│   ├── README.dataset.txt    # Roboflow dataset details
│   └── README.roboflow.txt   # Roboflow export metadata
├── weights/                  # Model weights (ONNX / INT8 / PyTorch)
│   └── .gitkeep
├── images/                   # Input test images
│   └── .gitkeep
├── crops/                    # Cropped output directory
│   └── .gitkeep
├── scripts/                  # Pipeline scripts
│   ├── train.py              # YOLO11s model training script
│   ├── export_onnx.py        # PyTorch -> ONNX conversion and alignment script
│   ├── quantize_onnx.py      # Graph optimization & Dynamic INT8 quantization script
│   └── extract_math.py       # Inference and auto-cropping pipeline script
├── .gitattributes            # Git line ending normalization
├── .gitignore                # Git exclusion rules
├── .venv/                    # Local virtual environment (ignored by Git)
├── LICENSE                   # MIT License legal terms
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```
