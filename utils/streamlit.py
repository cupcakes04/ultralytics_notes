from ultralytics import solutions
from pathlib import Path
import sys

script_path = Path(__file__).resolve().parent


# Make sure to run the file using command `streamlit run path/to/file.py`
def streamlit_app(model=script_path / "models/yolov8s-worldv2.pt"):

    inf = solutions.Inference(
        model=str(model),  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
    )

    inf.inference()

if __name__ == "__main__":
    # Check if a command-line argument was provided
    if len(sys.argv) > 1:
        # If an argument is provided, use it as the image path
        # sys.argv[0] is the script name, sys.argv[1] is the first argument
        model = sys.argv[1]
        print(f"Inserted model: {model}")
        streamlit_app(model=model)
    else:
        print('errored, please enter 1 arg only')