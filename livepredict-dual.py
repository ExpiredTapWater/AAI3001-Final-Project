import torch
import os
import cv2
import numpy as np
from matplotlib import cm
from depth_pro import create_model_and_transforms, load_rgb
from ultralytics import YOLO
import threading
import time

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

rgb_model_path = r"C:\Users\ChenYi\Documents\Github\AAI3001-Final-Project\3dprint_yolov11m\train2\weights\best.pt"
depth_model_path = r"C:\Users\ChenYi\Documents\Github\AAI3001-Final-Project\3dprint_depthpro_yolov11m\train\weights\best.pt"
rtsp = "rtsp://camera_username:camera_password@192.168.68.61:554/stream2"

def get_torch_device():
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

device = get_torch_device()
print(f"Device: {device}")
print("Loading Apple depth model....")
depthpro, transform = create_model_and_transforms(
    device=device,
    precision=torch.half,
)

depthpro.eval()

print("Loading YOLO models....")
rgb_model = YOLO(rgb_model_path)
depth_model = YOLO(depth_model_path)

# Shared variables between threads
latest_depth_map = None
frame_lock = threading.Lock()
processing = False


# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

# Check if the video capture is opened correctly
if not cap.isOpened():
    exit()

print("Starting")

def process_frame(frame):
    global latest_depth_map, processing
    processing = True
    
    # Convert frame to RGB (OpenCV uses BGR by default)
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to the expected input size of your model
    #image = cv2.resize(image, (640, 384))
    image = cv2.resize(frame, (640, 384))

    # Apply any necessary transformations
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension if needed

    # Run DepthPro inference
    try:
        prediction = depthpro.infer(input_tensor)
    except Exception as e:
        print(f"DepthPro inference error: {e}")
        processing = False
        return

    # Extract depth and compute inverse depth
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    inverse_depth_normalized = np.clip(inverse_depth_normalized, 0, 1)

    # Map inverse depth to color for visualization
    color_depth = cm.viridis(inverse_depth_normalized)
    color_depth = (color_depth[:, :, :3] * 255).astype(np.uint8)

    # Convert color_depth to BGR for OpenCV
    color_depth_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference on the original frame
    rgb_results = rgb_model.predict(image, imgsz=(384, 640), verbose=False)
    depth_results = depth_model.predict(image, imgsz=(384, 640), verbose=False)
    
    # Draw bounding boxes directly on the depth map
    for result in rgb_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        labels = result.boxes.cls.cpu().numpy()  # Class labels

        for (box, confidence, label) in zip(boxes, confidences, labels):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box)
            label_text = f"{rgb_model.names[int(label)]} {confidence:.2f}"

            # Draw the bounding box on the depth map
            cv2.rectangle(color_depth_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(
                color_depth_bgr,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
  
    # Draw bounding boxes directly on the depth map
    for result in depth_results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        labels = result.boxes.cls.cpu().numpy()  # Class labels

        for (box, confidence, label) in zip(boxes, confidences, labels):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box)
            label_text = f"{depth_model.names[int(label)]} {confidence:.2f}"

            # Draw the bounding box on the depth map
            cv2.rectangle(color_depth_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red box
            cv2.putText(
                color_depth_bgr,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    # Store the latest depth map
    with frame_lock:
        latest_depth_map = color_depth_bgr

    processing = False

    # Print statement to indicate that the model has updated the depth map
    print("Model updated: Depth map with bounding boxes computed.")


# Set the interval for processing frames (in seconds)
process_interval = 0.75  # Adjust this value as needed
last_process_time = 0

# Loop over frames from the video stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Start processing if not already processing and interval has passed
    if not processing and (current_time - last_process_time) >= process_interval:
        # Start a new thread for processing the frame
        threading.Thread(target=process_frame, args=(frame.copy(),)).start()
        last_process_time = current_time

    # Display the latest depth map if available
    with frame_lock:
        if latest_depth_map is not None:
            display_frame = latest_depth_map
        else:
            # Display a placeholder or the original frame until the depth map is ready
            display_frame = frame

    # Display the frame
    cv2.imshow('Estimated Depth Map', display_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
