from flask import Flask, Response
import cv2
import os

app = Flask(__name__)
# Environment setup for RTSP streaming
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

rtsp = "rtsp://camera_username:camera_password@192.168.68.61:554/stream1"
camera = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

# Initialize the webcam
#camera = cv2.VideoCapture(0)  # Use 0 for the default camera

def generate_frames():
    while True:
        # Read frames from the webcam
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame as an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Serve the video feed to the webpage
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
