import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("UItest_best.pt")

# Extract a single frame from the video
cap = cv2.VideoCapture("temp_video.mp4")
ret, frame = cap.read()
if ret:
    # Perform inference
    results = model.predict(frame, conf=0.3)
    print(results[0].boxes)  # Check detections
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow("Annotated Frame", annotated_frame)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
