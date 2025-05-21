import pathlib

# ⚠️ 添加这一行，让 yolov5 的 torch.hub 不再尝试使用 PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# 把 yolov5 的路径加到环境变量中（你要替换成你实际的 yolov5 路径）
FILE = Path(__file__).resolve()
YOLOV5_DIR = FILE.parents[1] / "yolov5"  # 假设 yolov5 在项目根目录下
sys.path.append(str(YOLOV5_DIR))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox


class YOLOv5Model:
    def __init__(self, weight_path, conf_thres=0.5, iou_thres=0.45,imgsz=320):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.model = None

    def load_model(self):
        try:
            self.model = DetectMultiBackend(self.weight_path, device=self.device)
            self.model.eval()
            self.stride = self.model.stride  # ✅ 正确写法，不加 .max()

            return True
        except Exception as e:
            print(f"[YOLOv5] 模型加载失败: {e}")
            return False

    def set_thresholds(self, conf, iou):
        self.conf_thres = conf
        self.iou_thres = iou

    # yolov5_model.py

    def predict(self, image_bgr):
        original_shape = image_bgr.shape[:2]  # h, w
        print(original_shape)

        img = letterbox(image_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        det = pred[0]
        print(det)

        # 🔥 坐标映射回原图尺寸
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_shape)
        print(det)
        return det

    def draw_results(self, frame, preds):
        print(f"[DEBUG] draw_results called. frame: {type(frame)}, preds: {type(preds)}")

        if preds is None or len(preds) == 0:
            print("[INFO] 无检测结果，跳过绘图")
            return frame

        preds = preds.cpu().numpy()  # 转为 numpy 格式
        preds = preds[preds[:, 4] >= self.conf_thres]
        h, w = frame.shape[:2]
        font_scale = max(min(w, h) / 1000.0, 0.5)  # 根据图像大小动态调整字体
        thickness = max(int(min(w, h) / 300), 1)  # 动态调整边框厚度

        for det in preds:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = float(det[4])
            cls = int(det[5])

            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)

        return frame





