import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

def export_model() -> Path:
    model_path = BASE_DIR / "weights" / "best.pt"

    if not model_path.exists():
        model_path = BASE_DIR / "runs" / "detect" / "math_lens" / "weights" / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found in 'weights/' or 'runs/': {model_path}")    
    
    model = YOLO(str(model_path))

    exported_onnx_str = model.export(format="onnx", imgsz=640, simplify=True)
    exported_path = Path(exported_onnx_str)

    target_onnx_path = BASE_DIR / "weights" / "best.onnx"
    target_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    if exported_path.resolve() != target_onnx_path.resolve():
        shutil.move(str(exported_path), str(target_onnx_path))

    print(f"ONNX model successfully saved to '{target_onnx_path}'")    
    return target_onnx_path

if __name__ == "__main__":
    try:
        export_model()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
