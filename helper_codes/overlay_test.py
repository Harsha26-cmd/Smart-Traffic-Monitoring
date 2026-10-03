import sys
import mss
import pygetwindow as gw
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

class Overlay(QtWidgets.QWidget):
    def __init__(self, target_window_title="Chrome"):
        super().__init__()
        self.target_window_title = target_window_title
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint|QtCore.Qt.WindowStaysOnTopHint|QtCore.Qt.Tool|QtCore.Qt.WindowTransparentForInput)
        self.sct = mss.mss()
        self.current_image = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        windows = gw.getWindowsWithTitle(self.target_window_title)
        if not windows:
            self.current_image = None
            self.update()
            return
        win = windows[0]
        monitor = {"top":win.top,"left":win.left,"width":win.width,"height":win.height}
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            return
        self.setGeometry(win.left,win.top,win.width,win.height)
        canvas = np.zeros((win.height,win.width,4),dtype=np.uint8)
        cv2.rectangle(canvas,(50,50),(win.width-50,win.height-50),(0,0,255,255),4)
        cv2.putText(canvas,"AI OVERLAY ACTIVE",(60,90),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0,255),2)
        h,w,ch=canvas.shape
        qImg=QtGui.QImage(canvas.data,w,h,ch*w,QtGui.QImage.Format_RGBA8888)
        self.current_image=qImg
        self.update()

    def paintEvent(self,event):
        if self.current_image:
            painter=QtGui.QPainter(self)
            painter.drawImage(0,0,self.current_image)

if __name__ == "__main__":
    app=QtWidgets.QApplication(sys.argv)
    overlay=Overlay("Chrome")
    overlay.show()
    sys.exit(app.exec_())
