import cv2
import os
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "QUIET"

VIDEO_PATH = "Videos/vehicle_and_helmet.mp4"
HELMET_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

OUTPUT_FOLDER = "violations_evidence"
NO_HELMET_FOLDER = "no_helmet_bikes"

HELMET_CLASS_IDS = {"helmet": 0, "no_helmet": 1}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(NO_HELMET_FOLDER, exist_ok=True)

def main():
    helmet_model = YOLO(HELMET_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    disp_width = 1024
    frame_count = 0
    cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        disp_height = int(frame.shape[0] * (disp_width / frame.shape[1]))
        frame = cv2.resize(frame, (disp_width, disp_height))
        frame_count += 1

        helmet_results = helmet_model(frame, imgsz=480, verbose=False)[0]
        canvas = frame.copy()

        for box in helmet_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
            cls_id = int(cls_id)

            if cls_id == HELMET_CLASS_IDS["no_helmet"]:
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(canvas, f"No Helmet {conf:.2f}", (x1,max(10,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                img_name = f"{NO_HELMET_FOLDER}/no_helmet_{frame_count}_{x1}_{y1}.jpg"
                h,w,_ = frame.shape
                crop_img = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                cv2.imwrite(img_name, crop_img)

            elif cls_id == HELMET_CLASS_IDS["helmet"]:
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(canvas, f"Helmet {conf:.2f}", (x1,max(10,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(canvas, "Press 'q' to quit.", (10,disp_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("Helmet Detection", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
