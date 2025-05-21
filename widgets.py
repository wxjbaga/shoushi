from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal



class ControlPanel(QFrame):

    conf_threshold_changed = pyqtSignal(float)
    iou_threshold_changed = pyqtSignal(float)
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.model_selector = QComboBox()
        self.model_selector.addItems(["YOLOv5s","FasterRCNN","SSD"])

        # self.compare_btn = QPushButton("多模型对比")
        self.conf_label = QLabel("置信度阈值: 0.5")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)

        self.iou_label = QLabel("IOU阈值: 0.45")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self.update_iou_label)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setColumnWidth(0, 85)  # 模型列
        self.metrics_table.setColumnWidth(1, 85)  # 准确率列
        self.metrics_table.setColumnWidth(2, 85)  # FPS列

        self.metrics_table.setHorizontalHeaderLabels(["模型", "准确率", "FPS"])
        self.metrics_table.setRowCount(3)

        layout.addWidget(QLabel("模型选择"))
        layout.addWidget(self.model_selector)
        # layout.addWidget(self.compare_btn)
        layout.addWidget(QLabel("参数设置"))
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_slider)
        layout.addWidget(self.iou_label)
        layout.addWidget(self.iou_slider)
        #layout.addWidget(QLabel("性能对比"))
        layout.addWidget(self.metrics_table)
        layout.addStretch()

    def update_conf_label(self, value):
        self.conf_label.setText(f"置信度阈值: {value / 100:.2f}")
        self.conf_threshold_changed.emit(value / 100)

    def update_iou_label(self, value):
        self.iou_label.setText(f"IOU阈值: {value / 100:.2f}")
        self.iou_threshold_changed.emit(value / 100)


class DisplayTabs(QTabWidget):
    def __init__(self):
        super().__init__()
        self.camera_tab = self.CameraTab()
        self.video_tab = self.VideoTab()
        self.image_tab = self.ImageTab()
        self.addTab(self.camera_tab, "摄像头")
        self.addTab(self.video_tab, "视频")
        self.addTab(self.image_tab, "图片")

    class CameraTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            self.video_label = QLabel("摄像头画面")
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("background: black;")
            self.start_btn = QPushButton("开启摄像头")
            # self.capture_btn = QPushButton("截图")
            # self.capture_btn.setEnabled(False)
            layout.addWidget(self.video_label)
            layout.addWidget(self.start_btn)
            # layout.addWidget(self.capture_btn)

    class VideoTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            self.video_label = QLabel("视频预览")
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setStyleSheet("background: black;")
            self.open_btn = QPushButton("打开视频")
            self.play_btn = QPushButton("播放")
            self.play_btn.setEnabled(False)
            layout.addWidget(self.video_label)
            layout.addWidget(self.open_btn)
            layout.addWidget(self.play_btn)

    class ImageTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            self.image_label = QLabel("图片预览")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("background: black;")
            self.open_btn = QPushButton("打开图片")
            # self.save_btn = QPushButton("保存结果")
            # self.save_btn.setEnabled(False)
            layout.addWidget(self.image_label)
            layout.addWidget(self.open_btn)
            # layout.addWidget(self.save_btn)