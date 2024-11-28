import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2

# Define the video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Load YOLO model
        self.model = YOLO("3dprint_yolov11m/train2/weights/best.pt")

    def transform(self, frame):
        # Convert the frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")

        # Perform YOLO inference
        results = self.model.predict(img, imgsz=640, conf=0.5)
        annotated_frame = results[0].plot()

        # Return the annotated frame
        return annotated_frame

# Streamlit live stream app
def main():
    st.title("Live Stream of 3D Printing")
    st.write("Watch the live feed from your connected camera with real-time failure detection.")
    
    # Start the live stream using webrtc_streamer
    webrtc_streamer(key="livestream", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
