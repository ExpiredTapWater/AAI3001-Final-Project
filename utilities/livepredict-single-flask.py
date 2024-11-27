import cv2
import os
from ultralytics import YOLO
from flask import Flask, Response

# Environment setup for RTSP streaming
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Paths and RTSP settings
model_path = r"C:\Users\ChenYi\Documents\Github\AAI3001-Final-Project\3dprint_yolov11m\train2\weights\best.pt"
rtsp = "rtsp://camera_username:camera_password@192.168.68.61:554/stream1"

# Load the YOLOv8 model
model = YOLO(model_path)

# Open the RTSP video stream
cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
#cap = cv2.VideoCapture(0)

# Flask application setup
app = Flask(__name__)

def generate_frames():
    while cap.isOpened():
        # Read a frame from the video stream
        success, frame = cap.read()
        if not success:
            break
        
        # Resize and perform YOLO inference
        resized_frame = cv2.resize(frame, (640, 384))
        results = model.predict(resized_frame, imgsz=(384, 640), verbose=False)

        # Annotate the frame with YOLO predictions
        annotated_frame = results[0].plot()

        # Encode the annotated frame to JPEG format
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_frame_bytes = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Serve the YOLO-annotated video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=False)
