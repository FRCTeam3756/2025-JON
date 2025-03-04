import os
import torch
import logging
from ultralytics import YOLO

script_name = os.path.splitext(os.path.basename(__file__))[0]

log_file = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    best_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logging("Running on device:", best_device)

    model: YOLO = YOLO('yolo11n.pt')
    model = model.half()
    
    try:
        model.train(
            data='vision_tracking/dataset/dataset.yaml',
            project='vision_tracking/runs',
            device=best_device,
            epochs=25, #75 is optimal but takes a long time
            imgsz=640, 
            batch=8, 
            workers=4,
            exist_ok=True,
            half=True
        )
    except Exception as e:
        logging("Error during training:", str(e))
        return

    try:
        results = model.val(device=best_device)
        logging("Validation Results:", results)
    except Exception as e:
        logging("Error during validation:", str(e))
        return

    try:
        model.export(format='onnx', half=True, device=best_device)
        model.export(format='engine', half=True, device=best_device)
    except Exception as e:
        logging("Error during export:", str(e))

if __name__ == '__main__':
    main()