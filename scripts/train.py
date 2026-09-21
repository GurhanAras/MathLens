import sys
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent

def train_model() -> Path:
    dataset_config = BASE_DIR / "dataset" / "data.yaml"
    output_dir = BASE_DIR / "runs" / "detect" / "math_lens"

    if not dataset_config.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at '{dataset_config}'")

    model = YOLO("yolo11s.pt")  

    results = model.train(
        data=str(dataset_config),  
        epochs=50,                 
        imgsz=640,                  
        batch=16,                   
        device=0,                  
        workers=2,                  
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        save=True,                  
        plots=True,
        patience=10,               
        mosaic=0.5,
        amp=True                    
    )
    
    print("Training completed successfully!")
    print(f"Results and weights saved to '{output_dir}'")
    return output_dir

if __name__ == "__main__":
    try:
        train_model()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
