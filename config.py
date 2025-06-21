# -*- coding: utf-8 -*-

from pathlib import Path
import sys

root_path = Path(__file__).parent.absolute()

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

SOURCES_LIST = ["Image", "Video"]
DETECTION_MODEL_LIST = []

MODEL_DIR = ROOT / "model"
MODEL_CONFIG = MODEL_DIR / "UECFOOD100" / "UECFOOD100.yaml"


def list_files(folder_path='./model', extension='.onnx'):
    """
    遍历文件夹下指定扩展名的文件

    Args:
        folder_path (str): 文件夹路径
        extension (str): 文件扩展名，如 '.txt', '.py'，None表示所有文件

    Returns:
        list: 符合条件的文件名字符串列表
    """
    try:
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            return []

        files = []
        for f in folder.iterdir():
            if f.is_file():
                if extension is None or f.suffix.lower() == extension.lower():
                    files.append(f.name)

        return sorted(files)

    except Exception as e:
        print(f"错误：{e}")
        return []
