# 带检测框的图像变换实践

import cv2


def draw_bboxes(img: cv2.Mat, onnx_outputs: list, model_img_size: int = 640):
    """绘制检测框

    Args:
        img (cv2.Mat): 原图
        onnx_outputs (list): [x, y, h, w, conf, class_id]
        model_img_size (int, optional): 模型输入图像尺寸. Defaults to 640.

    Returns:
        cv2.Mat: 绘制好的图像
    """
    res_img = img.copy()
    orig_h, orig_w = img.shape[:2]
    scale = min(model_img_size / orig_h, model_img_size / orig_w)
    new_h = int(scale * orig_h)
    new_w = int(scale * orig_w)
    padding_h = model_img_size - new_h
    padding_w = model_img_size - new_w
    padding_top = padding_h // 2
    padding_bottom = padding_h - padding_top
    padding_left = padding_w // 2
    padding_right = padding_w - padding_left

    for output in onnx_outputs:
        x, y, h, w, conf, classid = output
        h = int(h / scale)
        w = int(w / scale)
        x = int((x - padding_left) / scale)
        y = int((y - padding_top) / scale)

        cv2.rectangle(res_img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return res_img


def preprocess(img: cv2.Mat, model_img_size: int = 640):
    """将输入图像拓展为(model_img_size, model_img_size, 3)大小的图像

    Args:
        img (cv2.Mat): 原图
        model_img_size (int, optional): 图像尺寸. Defaults to 640.

    Returns:
        cv2.Mat: 拓展后的图像
    """
    h, w = img.shape[:2]
    scale = min(model_img_size / h, model_img_size / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    padding_h = model_img_size - new_h
    padding_w = model_img_size - new_w
    padding_top = padding_h // 2
    padding_bottom = padding_h - padding_top
    padding_left = padding_w // 2
    padding_right = padding_w - padding_left
    img.resize((new_h, new_w, 3))
    padded_img = cv2.copyMakeBorder(
        img,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded_img


if __name__ == "__main__":
    model_img_size = 640

    orig_img = cv2.imread("C:\\Users\\lybin\\Pictures\\UECFOOD-23.jpg")
    img = orig_img.copy()
    print(f"orig_img shape: {orig_img.shape}")

    padded_img = preprocess(orig_img)
    print(f"padded_img shape: {padded_img.shape}")
    x1, y1, x2, y2 = 60, 230, 520, 550
    cv2.rectangle(padded_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    onnx_output = [
        (
            x1,
            y1,
            (y2 - y1),
            (x2 - x1),
            0.6,
            5,
        )
    ]
    restore_img = draw_bboxes(orig_img, onnx_output)

    cv2.imshow("orig image", img)
    cv2.imshow("padded image", padded_img)
    cv2.imshow("restore image", restore_img)
    cv2.waitKey(2000)
