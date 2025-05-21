from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from widgets import *
from models.yolov5_model import YOLOv5Model
import queue
from thread_worker import VideoWorker
from CameraWorker import CameraWorker
import cv2  # 注意确保已经 import cv2
import os
import traceback
from frcnn import FRCNN
from ssd import SSD
from PIL import Image
import numpy as np
import threading
import torch
from PyQt5.QtCore import QTimer



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别系统 v1.0")
        self.setMinimumSize(1200, 800)
        self.video_thread = None

        self.cap = None  # 视频/摄像头对象
        self.timer = QTimer()  # 定时器，用于循环刷新帧
        self.video_path = None  # 视频文件路径


        # 初始化UI
        self.init_ui()
        self.init_models()
        self.connect_signals()
        self.write_index = 0  # 当前写入的行索引（0~2）

        self.camera_frame_queue = queue.Queue(maxsize=3)
        self.video_frame_queue = queue.Queue(maxsize=3)
        self.video_worker = VideoWorker(self.current_model, self.video_frame_queue)
        self.video_worker.result_ready.connect(self.display_result_frame)
        self.video_worker.performance_ready.connect(self.update_performance_table)

        self.camera_worker = CameraWorker(self.current_model, self.camera_frame_queue)
        self.camera_worker.result_ready.connect(self.display_result_frame)
    def init_ui(self):
        """初始化界面布局"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 左侧控制面板
        self.left_panel = ControlPanel()
        self.left_panel.setFixedWidth(300)  # 🔧 设置固定宽度，比如300px
        main_layout.addWidget(self.left_panel)

        # 右侧显示区域
        self.right_panel = DisplayTabs()
        main_layout.addWidget(self.right_panel, stretch=3)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪", 3000)



    def init_models(self):
        """初始化模型"""

        # 在 MainWindow 的 init_models 方法中

        self.models = {
            "YOLOv5s": YOLOv5Model("yolov5/weights/best.pt"),
            "FasterRCNN": FRCNN(),
            "SSD": SSD()
        }
        for model in self.models.values():
            model.load_model()
        self.current_model = self.models["YOLOv5s"]

    def reset_thread_worker(self, new_model):
        # 关闭旧线程
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()

        # 创建新线程（确保模型同步）
        self.video_worker = VideoWorker(new_model, self.video_frame_queue)
        self.video_worker.result_ready.connect(self.display_result_frame)
        self.camera_worker = CameraWorker(new_model, self.camera_frame_queue)
        self.camera_worker.result_ready.connect(self.display_result_frame)
        self.video_worker.performance_ready.connect(self.update_performance_table)
    def change_model(self, model_name):
        if model_name in self.models:
            self.current_model = self.models[model_name]
            self.reset_thread_worker(self.current_model)
            print(self.video_worker.model)
            self.status_bar.showMessage(f"已切换模型: {model_name}", 3000)


    def connect_signals(self):
        """连接信号槽"""
        # 模型选择
        self.left_panel.model_selector.currentTextChanged.connect(self.change_model)

        self.left_panel.conf_threshold_changed.connect(self.update_model_conf)
        self.left_panel.iou_threshold_changed.connect(self.update_model_iou)

        # 摄像头控制
        self.right_panel.camera_tab.start_btn.clicked.connect(self.start_camera_detection)

        # 视频控制
        self.right_panel.video_tab.open_btn.clicked.connect(self.open_video_dialog)

        # 图片控制
        self.right_panel.image_tab.open_btn.clicked.connect(self.open_image_dialog)
        self.right_panel.video_tab.play_btn.clicked.connect(self.play_video)


    def update_model_conf(self, value):
        print(f"[参数变化] 新置信度阈值: {value}")

        try:
            safe_value = max(0.01, min(0.99, float(value)))

            if isinstance(self.current_model, YOLOv5Model):
                self.current_model.set_thresholds(safe_value,self.current_model.iou_thres, )

            elif getattr(self.current_model, 'is_fasterrcnn', False) or getattr(self.current_model, 'is_ssd', False):
                if not hasattr(self, '_model_lock'):
                    self._model_lock = threading.Lock()
                with self._model_lock:
                    self.current_model.set_thresholds(safe_value,self.current_model.nms_iou )
                    print(f"[√] Faster R-CNN 新置信度已设置为: {safe_value}")
            QTimer.singleShot(300, self.refresh_results)
        except Exception as e:
            print(f"[错误] 更新置信度阈值时出错: {e}")

    def update_model_iou(self, value):
        print(f"[参数变化] 新IOU阈值: {value}")
        try:
            safe_value = max(0.01, min(0.99, float(value)))

            if isinstance(self.current_model, YOLOv5Model):
                self.current_model.set_thresholds(self.current_model.conf_thres, safe_value)

            elif getattr(self.current_model, 'is_fasterrcnn', False) or getattr(self.current_model, 'is_ssd', False):
                if not hasattr(self, '_model_lock'):
                    self._model_lock = threading.Lock()
                with self._model_lock:
                    self.current_model.set_thresholds(self.current_model.confidence, safe_value)
                    print(f"[√] Faster R-CNN 新IOU已设置为: {safe_value}")

            QTimer.singleShot(300, self.refresh_results)

        except Exception as e:
            print(f"[错误] 更新 IOU 阈值时出错: {e}")

    def open_video_dialog(self):
        """打开视频文件对话框"""
        options = QFileDialog.Options()
        video_file, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
            options=options)

        if video_file:
            if hasattr(self, "cap") and self.cap is not None:
                self.timer.stop()
                self.cap.release()
                self.video_worker.stop()
                self.video_frame_queue.queue.clear()
            self.video_path = video_file
            self.right_panel.video_tab.video_label.setText("视频加载成功")
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.status_bar.showMessage("视频文件打开失败", 3000)
                return

            # 检查分辨率，自动判断是否需要旋转
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(w,h)
            self.need_rotate = w < h  # 横屏拍摄视频需要旋转
            if not self.cap.isOpened():
                self.status_bar.showMessage("视频文件打开失败", 3000)
                return
            self.right_panel.video_tab.play_btn.setEnabled(True)
            self.status_bar.showMessage(f"已打开视频: {video_file}", 3000)

    def play_video(self):
        if not self.cap:
            return
        if not self.video_worker.isRunning():
            self.video_worker.running = True
            self.video_worker.start()
        # 断开旧的 timer 信号连接，避免重复连接
        try:
            self.timer.timeout.disconnect()
        except Exception:
            pass
        print("1")
        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(30)

    def update_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            self.video_worker.stop()
            return

        if self.video_frame_queue.qsize() < 3:  # 防止过多堆积
            print("[update_camera_frame] putting frame into queue")
            self.video_frame_queue.put(frame.copy())

    def display_result_frame(self, frame,source):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)

        if source == 'video':
            # 显示在视频区域的 QLabel 上
            pix = QPixmap.fromImage(img).scaled(self.right_panel.video_tab.video_label.size(), Qt.KeepAspectRatio)
            self.right_panel.video_tab.video_label.setPixmap(pix)
        elif source == 'camera':
            # 显示在摄像头区域的 QLabel 上
            pix = QPixmap.fromImage(img).scaled(self.right_panel.camera_tab.video_label.size(), Qt.KeepAspectRatio)
            self.right_panel.camera_tab.video_label.setPixmap(pix)


    # ========== 原有方法 ==========
    def start_camera_detection(self):
        btn = self.right_panel.camera_tab.start_btn

        if btn.text() == "开启摄像头":
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_bar.showMessage("无法打开摄像头", 3000)
                return
            # 启动推理线程
            if not self.camera_worker.isRunning():
                self.camera_worker.running = True
                self.camera_worker.start()
            # 切换按钮状态
            btn.setText("关闭摄像头")

            # 断开之前的定时器绑定，防止重复绑定
            try:
                self.timer.timeout.disconnect()
            except Exception:
                pass

            # 每30ms读取一帧送入推理线程
            self.timer.timeout.connect(self.update_camera_frame)
            self.timer.start(30)
            self.status_bar.showMessage("摄像头已开启", 2000)

        else:
            # 停止摄像头与线程
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self.camera_worker.stop()

            # 清空画面与状态
            self.right_panel.camera_tab.video_label.clear()
            btn.setText("开启摄像头")
            self.status_bar.showMessage("摄像头已关闭", 2000)

    def update_camera_frame(self):
        if not self.cap:
            print("[update_camera_frame] cap is None")
            return
        ret, frame = self.cap.read()
        if not ret:
            print("[update_camera_frame] frame not received")
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.camera_worker.stop()
            self.status_bar.showMessage("摄像头读取失败", 3000)
            return
        print("[update_camera_frame] got frame")

        print(self.camera_frame_queue.qsize())
        if self.camera_frame_queue.qsize() < 3:
            print("[update_camera_frame] putting frame into queue")
            self.camera_frame_queue.put(frame.copy())
        else:
            print("[update_camera_frame] queue full, skipping")
    def open_image_dialog(self):
        try:
            options = QFileDialog.Options()
            image_file, _ = QFileDialog.getOpenFileName(
                self, "选择图片文件", "",
                "图片文件 (*.jpg *.png *.bmp);;所有文件 (*)",
                options=options)

            if not image_file:
                return

            frame = cv2.imread(image_file)
            self.current_image = frame.copy()
            # 判断是否是使用的 Faster R-CNN 模型
            if hasattr(self.current_model, 'is_fasterrcnn') and self.current_model.is_fasterrcnn or hasattr(self.current_model, 'is_ssd') and self.current_model.is_ssd:
                # OpenCV 图像 -> PIL
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results, image_for_draw = self.current_model.predict(image_pil)
                frame = self.current_model.draw_results(image_for_draw, results)

                if isinstance(results, list) and len(results) > 0:
                    detections = results[0]
                    if isinstance(detections, torch.Tensor):
                        detections = detections.cpu().numpy()
                    if detections.size > 0:
                        if detections.shape[1] == 6:  # SSD/FasterRCNN 格式
                            # 有些是 [x1, y1, x2, y2, score, class_id]，有些是 [x1, y1, x2, y2, class_id, score]
                            if detections[0, 4] < 1 and detections[0, 5] >= 1:  # [x1, y1, x2, y2, score, class_id]
                                confidences = detections[:, 4]
                            else:  # [x1, y1, x2, y2, class_id, score]
                                confidences = detections[:, 5]
                            max_conf = float(confidences.max())
                        else:
                            max_conf = 0.0
                    else:
                        max_conf = 0.0
                else:
                    max_conf = 0.0

                # if not isinstance(frame, np.ndarray):
                #     frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            else:
                if frame is None:
                    QMessageBox.critical(self, "错误", "图片读取失败，可能是文件路径或格式错误")
                    return
                # === 模型预测与绘图 ===
                results = self.current_model.predict(frame)
                if isinstance(results, torch.Tensor):
                    result = results.cpu().numpy()  # 转成 numpy
                if len(result) > 0:
                    confidences = result[:, 4]  # 第 5 列是置信度
                    max_conf = float(confidences.max())
                else:
                    max_conf = 0.0
                if results is None:
                    QMessageBox.critical(self, "错误", "模型预测失败")
                    return
                # 传进去
                frame = self.current_model.draw_results(frame, results)

            if frame is None or not hasattr(frame, "shape"):
                QMessageBox.critical(self, "错误", "预测结果为空，无法显示")
                return
            # 获取模型名称
            if hasattr(self.current_model, 'is_fasterrcnn') and self.current_model.is_fasterrcnn:
                model_name = "Faster R-CNN"
            elif hasattr(self.current_model, 'is_ssd') and self.current_model.is_ssd:
                model_name = "SSD"
            else:
                model_name = "YOLOv5"
            # === 显示图像 ===
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.right_panel.image_tab.image_label.size(), Qt.KeepAspectRatio)

            self.right_panel.image_tab.image_label.setPixmap(pixmap)

            self.update_confidence_in_table(model_name,max_conf)
            self.status_bar.showMessage(f"已识别图片: {image_file}", 3000)

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[CRASH] 图片识别错误:\n{error_msg}")

            # 显示简化的错误信息给用户
            error_type = type(e).__name__
            error_location = traceback.extract_tb(e.__traceback__)[-1]  # 获取最后一个错误位置
            user_friendly_msg = (
                f"程序出错:\n"
                f"错误类型: {error_type}\n"
                f"错误位置: {error_location.filename} 第 {error_location.lineno} 行\n"
                f"错误详情: {str(e)}"
            )

            QMessageBox.critical(self, "异常", user_friendly_msg)

    def update_confidence_in_table(self, model_name: str,confidence):
        row = self.write_index
        self.left_panel.metrics_table.setItem(row, 0, QTableWidgetItem(model_name))
        item = QTableWidgetItem(f"{confidence:.3f}")
        # 假设置信度放左侧表格第0行第1列，调整索引对应你的表格
        self.left_panel.metrics_table.setItem(row, 1, item)
        self.write_index = (self.write_index + 1) % 3  # 循环使用三行

    def update_performance_table(self, fps, avg_conf):
        if hasattr(self.current_model, 'is_fasterrcnn') and self.current_model.is_fasterrcnn:
            model_name = "Faster R-CNN"
        elif hasattr(self.current_model, 'is_ssd') and self.current_model.is_ssd:
            model_name = "SSD"
        else:
            model_name = "YOLOv5"
        row = self.write_index
        self.left_panel.metrics_table.setItem(row, 0, QTableWidgetItem(model_name))
        self.left_panel.metrics_table.setItem(row, 2, QTableWidgetItem(f"{fps:.2f}"))  # 假设第2列是FPS
        self.left_panel.metrics_table.setItem(row, 1, QTableWidgetItem(f"{avg_conf:.2f}"))  # 假设第3列是平均置信度
        self.write_index = (self.write_index + 1) % 3  # 循环使用三行

    def refresh_results(self):
        if not hasattr(self, "current_image") or self.current_image is None:
            return
        try:
            frame = self.current_image.copy()
            # 处理 Faster R-CNN 模型
            if hasattr(self.current_model, 'is_fasterrcnn') and self.current_model.is_fasterrcnnor or hasattr(self.current_model, 'is_ssd') and self.current_model.is_ssd:
                # OpenCV 图像 -> PIL
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results, image_for_draw = self.current_model.predict(image_pil)
                if results is None:
                    QMessageBox.critical(self, "错误", "Faster R-CNN 模型预测失败")
                    return
                frame = self.current_model.draw_results(image_for_draw, results)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换回BGR保持一致性
            # 处理 YOLOv5 或其他模型
            else:
                self.current_results = self.current_model.predict(frame)

                if self.current_results is None:
                    QMessageBox.critical(self, "错误", "模型预测失败")
                    return

                frame = self.current_model.draw_results(frame, self.current_results)
            # 统一显示处理
            if frame is None or not hasattr(frame, "shape"):
                QMessageBox.critical(self, "错误", "预测结果为空，无法显示")
                return

            # 转换为RGB格式显示
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = display_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.right_panel.image_tab.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation)

            self.right_panel.image_tab.image_label.setPixmap(pixmap)

        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[CRASH] 刷新结果错误:\n{error_msg}")
            QMessageBox.critical(self, "错误", f"刷新结果时发生异常: {str(e)}")

