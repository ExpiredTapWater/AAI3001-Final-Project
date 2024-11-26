import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io
import os
import ntfy
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the YOLO11 model
def load_model():
    model_path = r"C:\Users\leena\OneDrive\Documents\Desktop\Deep Learning\Final Project\best.pt"
    model = YOLO(model_path)  # Load the YOLO11 model using Ultralytics
    return model

# Send notifications using ntfy
def send_ntfy_notification(topic, title, message):
    try:
        ntfy.notify(
            topic=topic,
            title=title,
            message=message,
            priority="high"
        )
    except Exception as e:
        st.error(f"Failed to send notification: {e}")

# Predict failure rate
def predict_failure_rate(image, model):
    # Perform inference
    results = model.predict(source=image, conf=0.3, save=True)  # Inference with confidence threshold 0.3

    # Extract bounding boxes and detection details
    detections = results[0].boxes  # Access the boxes attribute
    if detections:
        # Convert detections to a pandas DataFrame
        detection_data = []
        for box in detections:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]

            # Extract class and confidence
            cls = int(box.cls[0])  # Class index
            conf = box.conf[0].item()  # Confidence score

            # Get the class name from the names dictionary
            class_name = results[0].names[cls]

            # Append data for the table
            detection_data.append([class_name, conf, x1, y1, x2, y2])

        # Create a DataFrame for display
        detection_df = pd.DataFrame(detection_data, columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])
        return detection_df, results
    else:
        return None, results  # No detections found

# Live stream functionality
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Add custom processing here if needed
        return frame

# Main Streamlit app
def main():
    # Navigation between pages
    page = st.sidebar.selectbox("Navigation", ["Prediction", "Live Stream"])

    if page == "Prediction":
        st.title("3D Printing Failure Rate Prediction with YOLO11")

        # Load the model once
        model = load_model()

        # Upload Section for Images
        st.header("Upload an Image of the Print")
        image_file = st.file_uploader("Upload a .jpg image", type=["jpg"])

        if image_file:
            # Extract the name of the uploaded file (without the extension)
            input_filename = os.path.splitext(image_file.name)[0]

            # Display the uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Predict button
            if st.button("Predict Failure Rate"):
                with st.spinner("Analyzing the image..."):
                    # Perform failure rate prediction
                    detection_df, results = predict_failure_rate(image, model)

                    if detection_df is not None:
                        # Display results
                        st.success("Predictions complete!")
                        st.write(f"Detected {len(detection_df)} failures.")
                        
                        # Show detections as a DataFrame
                        st.dataframe(detection_df)

                        # Annotate and display the image with detections
                        annotated_image = results[0].plot()  # Annotate predictions on the image
                        
                        # Convert the annotated image to a format suitable for download
                        annotated_image_pil = Image.fromarray(annotated_image)  # Convert annotated image to PIL format
                        buf = io.BytesIO()
                        annotated_image_pil.save(buf, format="PNG")
                        buf.seek(0)

                        # Display the annotated image
                        st.image(annotated_image, caption="Prediction with Annotations", use_column_width=True)

                        # Send a notification if bad predictions are detected
                        failure_classes = detection_df["Class"].tolist()
                        send_ntfy_notification(
                            topic="AAI3001-FinalProj", # GUYS USE THIS TOPIC
                            title="Bad Predictions Detected",
                            message=f"{len(detection_df)} failures detected: {', '.join(failure_classes)}."
                        )

                        # Create a download button for the annotated image
                        st.download_button(
                            label="Download Prediction Image",
                            data=buf,
                            file_name=f"{input_filename}_prediction.png",  # Use input file name for the prediction
                            mime="image/png"
                        )
                    else:
                        st.warning("No failures detected in the image.")
    
    elif page == "Live Stream":
        st.title("Live Stream of 3D Printing")
        st.write("Watch the live feed from your connected camera.")
        
        # Start the live stream using streamlit-webrtc
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
