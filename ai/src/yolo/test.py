from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import json
import yaml
import torch

def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 읽어옵니다.

    Args:
        config_path: YAML 파일 경로

    Returns:
        : 설정 정보를 담은 Dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    if torch.cuda.is_available():
        device = 0
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS is available. Using MPS.")
    else:
        print("CUDA or MPS is not available. Using CPU.")
        device = "cpu"

    config = load_config("interface.yaml")

    output_file_path = Path(f"{config['output_dir']}/submission.csv")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config["class_mapping_file"], "r", encoding="utf-8") as f:
        class_map = json.load(f)

    reversed_map = {
        int(v): int(k)
        for k, v in class_map.items()
    }

    model = YOLO(config["model"])
    rows = []

    annotation_id = 1

    for image_path in sorted(Path(config["test_dir"]).glob("*")):
        image_id = int(image_path.stem)

        results = model.predict(
            source=image_path,
            imgsz=config["imgsz"],
            conf=config["conf"],
            iou=config["iou"],
            device=device,
            verbose=False
        )

        result = results[0]

        for box in result.boxes:
            cls = int(box.cls.item())
            score = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": reversed_map[cls],
                "bbox_x": round(x1),
                "bbox_y": round(y1),
                "bbox_w": round(x2 - x1),
                "bbox_h": round(y2 - y1),
                "score": score
            })

            annotation_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()