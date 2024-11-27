import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import os
import tempfile
import requests
import time 

# Global variable to track the last notification time
last_notification_time = 0 

# Load the YOLO model
def load_model():
    model_path = r"3dprint_yolov11m\train2\weights\best.pt"
    model = YOLO(model_path)  # Load the YOLO model
    return model

# Send notifications using ntfy
def send_ntfy_notification(topic, title, message):
    try:
        response = requests.post(
            f"https://ntfy.sh/{topic}",
            headers={"Title": title},
            data=message
        )
        if response.status_code == 200:
            st.success("Notification sent successfully!")
        else:
            st.error(f"Failed to send notification: {response.status_code}")
    except Exception as e:
        st.error(f"Error sending notification: {e}")

# Process video for YOLO detection
def process_video(video_path, model, confidence_threshold=0.5, iou_threshold=0.4, failure_confirmation_frames=3, cooldown_seconds=12):
    global last_notification_time

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Failed to open video file!")
        return

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a temporary file for saving the processed video
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        st.error("Failed to initialize video writer!")
        return

    # Placeholder for dynamic information update
    info_placeholder = st.empty()

    # Progress bar placeholder
    progress_bar = st.progress(0)

    # Placeholder for displaying the video frames in Streamlit
    video_placeholder = st.empty()

    info_placeholder.write(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    frame_count = 0
    failure_detected_in_last_frames = 0  # Counter for consecutive frames with failure detected

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

        # Resize the annotated frame back to original dimensions for consistent video output
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

        detections = results[0].boxes

        failure_detected = False
        if detections:
            for box in detections:
                cls = int(box.cls[0])
                class_name = results[0].names[cls]
                confidence = box.conf[0].item()

                # Apply confidence threshold and filter by class
                if class_name == "purge" and confidence >= confidence_threshold:
                    failure_detected = True
                    break

        # Update the failure detection counter
        if failure_detected:
            failure_detected_in_last_frames += 1
        else:
            failure_detected_in_last_frames = 0

        # Send notification if failure is detected in consecutive frames and cooldown period has passed
        current_time = time.time()
        if (
            failure_detected_in_last_frames >= failure_confirmation_frames
            and current_time - last_notification_time >= cooldown_seconds
        ):
            send_ntfy_notification(
                topic="AAI3001-FinalProj",
                title="3D Printing Failure Detected",
                message=f"Failure detected consistently in {failure_detected_in_last_frames} consecutive frames. Current frame: {frame_count}."
            )
            # Update the last notification time
            last_notification_time = current_time

        # Save the annotated frame to the output video
        out.write(annotated_frame)

        # Update the progress bar
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Display the annotated frame in the Streamlit app
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        # Update the information (frame count / total frames)
        info_placeholder.write(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Processed Frames: {frame_count}/{total_frames}")

    cap.release()
    out.release()
    st.success("Video processing complete.")

    # Provide a download link for the processed video
    if os.path.exists(temp_output_path):
        with open(temp_output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

def main():
    st.title("3D Printing Failure Detection and Prediction")
    model = load_model()

    # Upload Section for Images and Videos
    st.header("Upload an Image or Video for Prediction")
    file = st.file_uploader("Upload a .jpg, .png, or .mp4 file", type=["jpg", "png", "mp4"])

    # Create placeholders for content
    display_placeholder = st.empty()  # Unified placeholder for both images
    download_button_placeholder = st.empty()  # Placeholder for the download button

    if file:
        file_extension = os.path.splitext(file.name)[-1].lower()

        # Process Image
        if file_extension in [".jpg", ".png"]:
            # Clear any previous download button
            download_button_placeholder.empty()

            image = Image.open(file)

            # Perform YOLO inference
            results = model.predict(source=image, conf=0.3)
            annotated_image = results[0].plot()

            # Display uploaded image first, then the annotated image
            with display_placeholder.container():
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.image(annotated_image, caption="Prediction with Annotations", use_container_width=True)

            # Check for detections and send notification if "failure" is detected
            detections = results[0].boxes
            failure_detected = False
            if detections:
                for box in detections:
                    cls = int(box.cls[0])
                    class_name = results[0].names[cls]
                    confidence = box.conf[0].item()
                    if class_name == "failure" and confidence >= 0.3:
                        failure_detected = True
                        st.write(f"Detected {class_name} with confidence {confidence:.2f}")
                        send_ntfy_notification(
                            topic="AAI3001-FinalProj",
                            title="3D Printing Failure Detected",
                            message="Failure detected in the uploaded image."
                        )
                        break

            if not failure_detected:
                st.info("No failures detected in the image.")

            # Save the annotated image to a temporary file
            temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            annotated_pil_image.save(temp_image_path, format="JPEG")

            # Provide a download button for the annotated image
            with open(temp_image_path, "rb") as f:
                download_button_placeholder.download_button(
                    label="Download Annotated Image",
                    data=f,
                    file_name="annotated_image.jpg",
                    mime="image/jpeg"
                )

        # Process Video
        elif file_extension == ".mp4":
            # Clear display and download button
            display_placeholder.empty()
            download_button_placeholder.empty()

            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_video_path, "wb") as f:
                f.write(file.read())

            # Process video
            process_video(temp_video_path, model)

if __name__ == "__main__":
    main()
