import cv2
import numpy as np
import time
import pygame
import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Driver Drowsiness Detection System")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #333333;
                color: #FFFFFF;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3E8E41;
            }
            QLabel#warning_label {
                color: #FF0000;
                font-size: 20px;
                font-weight: bold;
            }
        """)

        # Label for camera 
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setStyleSheet("background-color: black;")

        # Label for warning text
        self.warning_label = QtWidgets.QLabel()
        self.warning_label.setObjectName("warning_label")

        # Open/Close camera button
        self.camera_button = QtWidgets.QPushButton("Open Camera")
        self.camera_button.clicked.connect(self.toggle_camera)

        # Layouts
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.camera_label)
        layout.addWidget(self.warning_label)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.camera_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.drowsy_time = 5
        self.reset()

        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm.wav")

    def reset(self):
        self.is_camera_open = False
        self.is_drowsy = False
        self.show_warning = False
        self.start_time = None

    def play_alarm(self):
        if not pygame.mixer.get_busy():
            self.alarm_sound.play(-1)

    def stop_alarm(self):
        self.alarm_sound.stop()

    def detect_sleep(self, frame):
        eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            roi_gray = gray[ey:ey+eh, ex:ex+ew]
            roi_color = frame[ey:ey+eh, ex:ex+ew]
            circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                return False
        return True

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        sleep = self.detect_sleep(frame)

        if sleep:
            if not self.is_drowsy:
                if time.time() - self.start_time >= self.drowsy_time:
                    self.is_drowsy = True
                    self.show_warning = True
                    self.warning_label.setText("Attention! The driver is sleeping!")
                    self.play_alarm()
        else:
            self.start_time = time.time()
            self.is_drowsy = False
            self.stop_alarm()
            self.show_warning = False
            self.warning_label.clear()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap)

    def toggle_camera(self):
        if not self.is_camera_open:
            self.cap = cv2.VideoCapture(0)
            self.start_time = time.time()
            self.timer.start(30)
            self.camera_button.setText("Close Camera")
            self.is_camera_open = True
        else:
            self.timer.stop()
            self.cap.release()
            self.camera_label.clear()
            self.warning_label.clear()
            self.stop_alarm()
            self.reset()
            self.camera_button.setText("Open Camera")
            self.is_camera_open = False

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())