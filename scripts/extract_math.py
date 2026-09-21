import sys
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def crop_math_expressions(
    image_path: Union[str, Path],
    model: YOLO,
    output_dir: Union[str, Path] = BASE_DIR / "crops",
    conf_threshold: float = 0.60,
    padding: int = 5,
    min_size: int = 12
) -> List[Path]:
    
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.is_file():
        print(f"Error: Image file not found -> '{image_path.name}'")
        return []

    img: Optional[np.ndarray] = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Failed to read image -> '{image_path.name}'")
        return []

    results = model(img, conf=conf_threshold, verbose=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    h_img, w_img = img.shape[:2]
    base_name = image_path.stem
    count = 0

    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            continue

        boxes_list = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())

            boxes_list.append({
                'coords': (x1, y1, x2, y2),
                'cls_id': cls_id,
                'class_name': model.names[cls_id]
            })

        sorted_boxes = sorted(
            boxes_list, 
            key=lambda b: (b['coords'][1] // 20, b['coords'][0])
        )

        for item in sorted_boxes:
            class_name = item['class_name']
            x1, y1, x2, y2 = item['coords']

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_img, x2 + padding)
            y2 = min(h_img, y2 + padding)

            crop_w, crop_h = x2 - x1, y2 - y1
            if crop_w < min_size or crop_h < min_size:
                continue

            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            count += 1
            output_file = output_dir / f"{base_name}-{class_name}-{count:02d}.png"

            cv2.imwrite(str(output_file), cropped)
            saved_paths.append(output_file)
            print(f"Saved [{class_name}] ({crop_w}x{crop_h}px): {output_file.name}")

    if not saved_paths:
        print(f"No elements detected in {image_path.name}")

    return saved_paths


def process_images_folder(
    input_dir: Union[str, Path] = BASE_DIR / "images",
    output_dir: Union[str, Path] = BASE_DIR / "crops",
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.60
) -> List[Path]:

    input_dir = Path(input_dir)

    if model_path is None:
        model_path = BASE_DIR / "weights" / "best_int8.onnx"
        if not model_path.exists():
            model_path = BASE_DIR / "weights" / "best.onnx"
        if not model_path.exists():
            model_path = BASE_DIR / "weights" / "best.pt"
        if not model_path.exists():
            model_path = BASE_DIR / "runs" / "detect" / "math_lens" / "weights" / "best.pt"
    else:
        model_path = Path(model_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Image directory not found -> '{input_dir}'")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found in 'weights/' or 'runs/': '{model_path}'")

    print(f"Loading model: {model_path.name}")
    model = YOLO(str(model_path), task="detect")    

    image_files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    
    if not image_files:
        print(f"[!] No valid image files found in '{input_dir}' directory.")
        return []

    print(f"Processing a total of {len(image_files)} images...\n" + "-" * 50)

    total_saved: List[Path] = []
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        saved = crop_math_expressions(
            image_path=img_path,
            model=model,  
            output_dir=output_dir,
            conf_threshold=conf_threshold
        )
        total_saved.extend(saved)

    print("-" * 50)
    print(f"Processing complete! Total elements cropped: {len(total_saved)}")
    return total_saved


if __name__ == "__main__":
    try:
        process_images_folder(conf_threshold=0.60)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
