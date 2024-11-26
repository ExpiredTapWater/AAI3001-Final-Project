import cv2
import os
from ultralytics import YOLO

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

model_path = r"./runs/detect/train3/weights/best.pt"
video_path = r"C:\Users\ChenYi\Desktop\CV\datasets\videos\video_timelapse.mp4"
rtsp = "rtsp://camera_username:camera_password@192.168.68.60:554/stream1"

# Load the YOLOv8 model
model = YOLO(model_path)
#model = YOLO("yolo11n.yaml")

# Open the video file
#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        #results = model.(frame)
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()