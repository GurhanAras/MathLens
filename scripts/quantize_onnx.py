import sys
from pathlib import Path
from typing import Union
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

BASE_DIR = Path(__file__).resolve().parent.parent


def quantize_model(
    input_path: Union[str, Path] = BASE_DIR / "weights" / "best.onnx",
    output_path: Union[str, Path] = BASE_DIR / "weights" / "best_int8.onnx"
) -> Path:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Source ONNX model not found at '{input_path}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_preprocessed_path = output_path.parent / f"{input_path.stem}_preprocessed.onnx"

    print("Running graph optimization & shape inference...")

    quant_pre_process(
        input_model_path=str(input_path),
        output_model_path=str(temp_preprocessed_path),
        skip_symbolic_shape=True
    )

    print("Applying dynamic INT8 quantization...")

    quantize_dynamic(
        model_input=str(temp_preprocessed_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8
    )

    if temp_preprocessed_path.exists():
        temp_preprocessed_path.unlink()

    original_size = input_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)

    print("Dynamic INT8 quantization complete!")
    print(f"Original ONNX (FP32) : {original_size:.2f} MB")
    print(f"Quantized ONNX (INT8): {quant_size:.2f} MB")
    print(f"Quantized model saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    try:
        quantize_model()
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
