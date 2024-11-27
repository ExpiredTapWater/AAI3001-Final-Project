import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import tempfile
import requests

# Load the YOLO model
def load_model():
    model_path = r"3dprint_yolov11m\train2\weights\best.pt"
    model = YOLO(model_path)  # Load the YOLO model
    return model

# # Send notifications using ntfy
# def send_ntfy_notification(topic, title, message):
#     try:
#         response = requests.post(
#             f"https://ntfy.sh/{topic}",
#             headers={"Title": title},
#             data=message
#         )
#         if response.status_code == 200:
#             st.success("Notification sent successfully!")
#         else:
#             st.error(f"Failed to send notification: {response.status_code}")
#     except Exception as e:
#         st.error(f"Error sending notification: {e}")

# Process video for YOLO detection
def process_video(video_path, model, confidence_threshold=0.5, iou_threshold=0.4):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Failed to open video file!")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Display video properties
    st.info(f"Video Loaded: {os.path.basename(video_path)}")
    st.write(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    # Placeholder for displaying the video frames in Streamlit
    video_placeholder = st.empty()

    frame_count = 0
    detection_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize frame for YOLO model
        resized_frame = cv2.resize(frame, (640, 384))

        # Perform YOLO inference
        results = model.predict(resized_frame, imgsz=(384, 640), conf=confidence_threshold, iou=iou_threshold)

        # Annotate the frame with detections if available
        annotated_frame = results[0].plot()
        detections = results[0].boxes

        if detections:
            for box in detections:
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                confidence = box.conf[0].item()

                # Apply confidence threshold and filter by class
                if class_name == "failure" and confidence >= confidence_threshold:
                    st.write(f"Frame {frame_count}: Detected {class_name} with confidence {confidence:.2f}")
                    detection_history.append(True)
                else:
                    detection_history.append(False)

        # Display the annotated frame in the Streamlit app
        video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        # # Confirm failure only if detected in 3 consecutive frames
        # if len(detection_history) >= 3 and sum(detection_history[-3:]) >= 3:
        #     send_ntfy_notification(
        #         topic="AAI3001-FinalProj",
        #         title="3D Printing Failure Detected",
        #         message=f"Failure detected consistently around frame {frame_count}."
        #     )
        #     detection_history = []  # Reset after notification

    cap.release()
    st.success("Video processing complete.")


# Main Streamlit app
def main():
    # Navigation between pages
    page = st.sidebar.selectbox("Navigation", ["Prediction", "Livestream"])

    if page == "Prediction":
        st.title("3D Printing Failure Rate Prediction with YOLO11")
        model = load_model()

        # Upload Section for Images
        st.header("Upload an Image of the Print")
        image_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "png"])

        if image_file:
            # Display uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform YOLO inference
            results = model.predict(source=image, conf=0.3)
            annotated_image = results[0].plot()

            # Display annotated image
            st.image(annotated_image, caption="Prediction with Annotations", use_column_width=True)

    elif page == "Livestream":
        st.title("3D Printing Failure Detection - Video")
        model = load_model()

        # Upload Section for Videos
        video_file = st.file_uploader("Upload a video file", type=["mp4"])

        if video_file:
            # Save uploaded video to a temporary file
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            # Process video
            process_video(temp_video_path, model)

if __name__ == "__main__":
    main()
