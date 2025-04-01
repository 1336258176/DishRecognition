# -*- coding: utf-8 -*-

from pathlib import Path
import sys

root_path = Path(__file__).parent.absolute()

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

SOURCES_LIST = ["Image", "Video"]
DETECTION_MODEL_LIST = ['yolov5s.onnx', 'yolov5s_sim.onnx']

MODEL_DIR = ROOT / "model"
MODEL_CONFIG = MODEL_DIR / "UECFOOD100" / "UECFOOD100.yaml"
