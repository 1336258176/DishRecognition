# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml
import logging
import random
import torchvision
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def preprocess(img, target_size=640):
    """
    图像预处理函数
    :param img: 输入图像（BGR格式）
    :param target_size: 模型输入尺寸
    :return:
        processed_img: 预处理后的numpy数组(1,3,H,W)
        scale: 缩放比例 (原图尺寸到模型输入尺寸的比例)
        pad: 填充尺寸 (top, bottom, left, right)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    scale = min(target_size / max(h, w), target_size / min(h, w))

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    processed_img = padded.astype(np.float32) / 255.0
    processed_img = processed_img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW

    return processed_img, scale, (top, left)


def postprocess(prediction, scale, pad, conf_thres=0.25, iou_thres=0.45, max_det=100):
    """
    后处理函数（包含NMS）
    :param prediction: ONNX模型输出的预测结果
    :param scale: 预处理时的缩放比例
    :param pad: 预处理时的填充尺寸 (top, left)
    :param conf_thres: 置信度阈值
    :param iou_thres: NMS的IOU阈值
    :param max_det: 用于控制最大检测数量
    :return: 检测结果列表 [x1, y1, x2, y2, conf, cls]
    """
    pred = np.concatenate(prediction, axis=1)
    x = torch.from_numpy(pred)

    boxes = x[..., :4]
    scores = x[..., 4:5] * x[..., 5:]

    max_scores, max_indices = torch.max(scores, dim=-1)
    mask = max_scores > conf_thres

    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = max_indices[mask]

    boxes = cxcywh_to_xyxy(boxes)

    keep = torchvision.ops.nms(boxes, scores, iou_thres)
    keep = keep[:max_det]

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes = scale_coords(boxes, scale, pad)

    return torch.cat([boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)], dim=1).cpu().numpy()


def draw_img(img, detections, class_names):
    """
    绘制检测结果到原图
    :param img: 原始图像（BGR格式）
    :param detections: 后处理得到的检测结果数组
    :param class_names: 类别名称列表
    :return: 绘制后的图像（BGR格式）
    """
    # 生成颜色映射
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        color = colors[int(cls_id)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{class_names[int(cls_id)]} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img


def cxcywh_to_xyxy(boxes):
    """将(cx, cy, w, h)格式转换为(x1, y1, x2, y2)格式"""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def scale_coords(boxes, scale, pad):
    """将框坐标转换回原始图像尺寸"""
    top, left = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    return boxes


def getClassesName(config_path: str):
    classes_name = []
    with open(config_path, mode='r', encoding='utf-8') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    classes = config_file.get('names')
    for idx, _ in enumerate(classes):
        classes_name.append(classes[idx])
    return classes_name


class Inference(object):
    def __init__(self, onnx_path: str):
        """检查onnx模型并初始化onnx"""
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            logger.error("Model incorrect!")

        options = ort.SessionOptions()
        options.enable_profiling = True
        self.onnx_session = ort.InferenceSession(onnx_path)
        # logger.info(f"onnx providers: {self.onnx_session.get_providers()}")

    def inference(self, image, conf_thres: float, iou_thres: float):
        """
        进行推理
        :param image: 输入图像
        :param conf_thres: 置信度阈值
        :param iou_thres: NMS阈值
        :return 检测结果列表 [x1, y1, x2, y2, conf, cls]
        """
        processed_img, scale, pad = preprocess(image)
        inputs = {self.onnx_session.get_inputs()[0].name: processed_img}
        outputs = self.onnx_session.run(None, inputs)
        detections = postprocess(outputs, scale, pad, conf_thres, iou_thres)
        return detections
