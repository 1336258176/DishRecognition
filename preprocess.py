# -*- coding: utf-8 -*-

import os
import shutil
import random
import yaml

from PIL import Image
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DatasetName = "UECFOOD100"
Size = (640, 640)


def read_bb_info(folder):
    bounding_box = {}
    if os.path.isdir(folder):
        bb_info_file = os.path.join(folder, "bb_info.txt")
        if os.path.isfile(bb_info_file):
            with open(bb_info_file, "r") as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    img_id = int(parts[0])
                    boxes = [
                        (
                            int(parts[i]),
                            int(parts[i + 1]),
                            int(parts[i + 2]),
                            int(parts[i + 3]),
                        )
                        for i in range(1, len(parts), 4)
                    ]
                    bounding_box[img_id] = boxes
    return bounding_box


def read_img_size(folder):
    img_size = {}
    for img_file in os.listdir(folder):
        if img_file.endswith(".jpg"):
            img_id = int(img_file[:-4])
            img = Image.open(os.path.join(folder, img_file))
            img_size[img_id] = img.size
            img.close()
    return img_size


def gen_yolo_label(input_folder, output_folder):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            img_sizes = read_img_size(folder_path)
            bb_infos = read_bb_info(folder_path)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.isfile(img_path) and img_name.endswith(".jpg"):
                    img_id = int(img_name[:-4])
                    img_width, img_height = img_sizes[img_id]
                    bb_size = bb_infos[img_id]
                    yolo_label_path = os.path.join(
                        output_folder, "labels", f"{img_id}.txt"
                    )
                    with open(yolo_label_path, "a") as f:
                        for box in bb_size:
                            category_id = folder_name
                            x_center = (box[0] + box[2]) / (2 * img_width)
                            y_center = (box[1] + box[3]) / (2 * img_height)
                            box_width = (box[2] - box[0]) / img_width
                            box_height = (box[3] - box[1]) / img_height
                            f.write(
                                f"{category_id} {x_center:.16f} {y_center:.16f} {box_width:.16f} {box_height:.16f}\n"
                            )


def copy_imgs(input_folder, output_folder):
    output_path = os.path.join(output_folder, "images")
    os.makedirs(output_path, exist_ok=True)
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".jpg"):
                    shutil.copy(os.path.join(folder_path, file_name), output_path)


def split_dataset(input_folder, output_folder, train_ratio=0.8):
    os.makedirs(os.path.join(output_folder, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels", "val"), exist_ok=True)

    image_folder = os.path.join(input_folder, "images")
    label_folder = os.path.join(input_folder, "labels")
    filenames = os.listdir(image_folder)
    random.shuffle(filenames)
    num_train = int(len(filenames) * train_ratio)

    for i, filename in enumerate(filenames):
        if i < num_train:
            shutil.move(os.path.join(image_folder, filename), os.path.join(output_folder, "images", "train"))
            shutil.move(os.path.join(label_folder, filename[:-4] + ".txt"),
                        os.path.join(output_folder, "labels", "train"))
        else:
            shutil.move(os.path.join(image_folder, filename), os.path.join(output_folder, "images", "val"))
            shutil.move(os.path.join(label_folder, filename[:-4] + ".txt"),
                        os.path.join(output_folder, "labels", "val"))


def convert_category_to_yaml(category_file, yaml_file):
    categories = {}
    with open(category_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            if line.strip():
                id_val, name = line.strip().split('\t')
                categories[int(id_val)] = name

    names_dict = {i: name for i, name in sorted(categories.items())}

    yaml_data = {
        'path': '../UECFOOD100-yolo',
        'train': 'images/train',
        'val': 'images/val',
        'test': ' ',
        'names': names_dict,
    }

    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"YAML generated: {yaml_file}")


if __name__ == "__main__":
    input_path = DatasetName
    output_path = DatasetName + "-yolo"
    category_file = DatasetName + "/category.txt"
    yaml_file = DatasetName + ".yaml"

    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels"), exist_ok=True)
    logger.info("Generating yolo labels...")
    gen_yolo_label(input_path, output_path)
    logger.info("Copying yolo images...")
    copy_imgs(input_path, output_path)
    logger.info("splitting dataset...")
    split_dataset(output_path, output_path)
    # logger.info("Generating yolo config file...")
    # convert_category_to_yaml(category_file, yaml_file)
    logger.warning("done!")
