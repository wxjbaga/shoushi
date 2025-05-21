import sys
import qdarkstyle
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui_main import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())