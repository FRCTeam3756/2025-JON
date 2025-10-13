import os
import re
import platform
import torch
from ultralytics import YOLO
import shutil


def get_system_identifier():
    system = platform.system().lower()
    arch = platform.machine().lower()
    if arch in ["amd64"]:
        arch = "x86_64"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_name = re.sub(r'[^a-zA-Z0-9]', '', gpu_name).lower()

    return f"{system}_{arch}_{gpu_name}"


def main():
    model = YOLO("src/vision/weights/best.pt")
    sys_id = get_system_identifier()
    
    output_dir = "src/vision/weights"
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"best_{sys_id}.engine"
    output_path = os.path.join(output_dir, output_name)

    print(f"Exporting model as: {output_path}")

    export_path = model.export(format='engine', device='cuda', half=True)

    if os.path.exists(export_path):
        shutil.move(export_path, output_path)
        print(f"Model exported successfully: {output_name}")
    else:
        print(f"Warning: export path {export_path} does not exist!")


if __name__ == "__main__":
    main()
