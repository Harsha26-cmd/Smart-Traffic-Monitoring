import cv2
import numpy as np
import mss
import pygetwindow as gw

def capture_window(window_title="Chrome"):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window with title '{window_title}' not found.")
        return
    win = windows[0]
    with mss.mss() as sct:
        cv2.namedWindow("Captured Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captured Window",800,600)
        while True:
            monitor={"top":win.top,"left":win.left,"width":win.width,"height":win.height}
            if monitor["width"]>0 and monitor["height"]>0:
                frame=np.array(sct.grab(monitor))
                frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
                cv2.imshow("Captured Window",frame)
            if cv2.waitKey(1)&0xFF==ord("q"):
                break
    cv2.destroyAllWindows()

if __name__=="__main__":
    capture_window("Chrome")
