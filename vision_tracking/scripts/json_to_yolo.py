import json
import os
from glob import glob

class_to_id = {
    "class1": 0,
    "class2": 1,
}

labelme_dir = "path/to/labelme/json/"
output_dir = "path/to/output/yolo/"

os.makedirs(output_dir, exist_ok=True)

for json_file in glob(os.path.join(labelme_dir, "*.json")):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    yolo_annotations = []
    
    for shape in data["shapes"]:
        label = shape["label"]
        points = shape["points"]
        
        if label not in class_to_id:
            continue
        
        class_id = class_to_id[label]
        normalized_points = [
            f"{x / image_width:.6f} {y / image_height:.6f}" for x, y in points
        ]
        
        yolo_line = f"{class_id} " + " ".join(normalized_points)
        yolo_annotations.append(yolo_line)
    
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".txt")
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

print("Conversion complete!")
