# utils/thread_worker.py
import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
import queue
import time

from PyQt5.uic.properties import QtCore


class VideoWorker(QThread):
    result_ready = pyqtSignal(object,str)

    performance_ready = pyqtSignal(float, float)  # fps, avg_conf
    def __init__(self, model, frame_queue, interval=0.03):
        super().__init__()
        self.model = model
        self.frame_queue = frame_queue
        self.result_queue = queue.Queue()
        self.running = False
        self.interval = interval


    def run(self):
        print("[Thread] prediction thread started")
        frame_count = 0
        start_time = time.time()
        total_confidence = 0.0
        total_detections = 0
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None or frame.size == 0:
                    continue
                frame = frame.copy()
                print("[Thread] got frame")
                if hasattr(self.model, 'is_fasterrcnn') and self.model.is_fasterrcnn or hasattr(
                        self.model, 'is_ssd') and self.model.is_ssd:
                    # OpenCV 图像 -> PIL
                    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).copy()
                    results, image_for_draw = self.model.predict(image_pil)

                    if isinstance(results, list) and len(results) > 0:
                        detections = results[0]
                        if isinstance(detections, list):
                            detections = np.array(detections)

                        if detections.size > 0:
                            if detections.shape[1] == 6:
                                # 判断格式：[x1, y1, x2, y2, score, class_id] 或 [x1, y1, x2, y2, class_id, score]
                                if detections[0, 4] < 1 and detections[0, 5] >= 1:
                                    confidences = detections[:, 4]
                                else:
                                    confidences = detections[:, 5]
                                total_confidence += confidences.sum()
                                total_detections += len(confidences)
                                print(total_confidence)
                                print(total_detections)
                    frame = self.model.draw_results(image_for_draw.copy(), results)
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame)  # PIL -> np.ndarray (RGB)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # frame=frame.copy()
                    results = self.model.predict(frame)
                    frame = self.model.draw_results(frame, results)
                    if isinstance(results, torch.Tensor):
                        results = results.cpu().numpy()

                    if results is not None and len(results) > 0 and results.shape[1] >= 5:
                        confidences = results[:, 4]  # 第 5 列是置信度
                        total_confidence += confidences.sum()
                        total_detections += len(confidences)

                frame_count += 1
                self.result_ready.emit(frame,"video")
            time.sleep(self.interval)
        end_time = time.time()
        total_time = end_time - start_time
        if total_time > 0 and frame_count > 0:
            fps = frame_count / total_time
            print(f"[Thread] total frames: {frame_count}, total time: {total_time:.2f}s, FPS: {fps:.2f}")
        else:
            print("[Thread] insufficient data for FPS calculation")
        if total_detections > 0:
            print("haha")
            avg_conf = total_confidence / total_detections
            print(avg_conf)
        else:
            avg_conf = 0.0
        self.performance_ready.emit(fps, avg_conf)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
