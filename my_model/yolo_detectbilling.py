import os
import sys
import argparse
import glob
import time
import sqlite3
import requests
import threading

import cv2
import csv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
from deep_sort_realtime.deepsort_tracker import DeepSort


import torch
import torchvision.models as models

from sklearn.metrics.pairwise import cosine_similarity




# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), USB camera ("usb0"), or \
                    phone camera stream ("http://<phone_ip>:<port>/video")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--api_endpoint', help='API endpoint to send detected object details', required=True)
parser.add_argument('--db_path', help='Path to SQLite database file containing inventory', required=True)
parser.add_argument('--conf_thresh', help='Confidence threshold for matching detected objects (example: "0.85")',
                    default=0.43)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
api_endpoint = args.api_endpoint
db_path = args.db_path
conf_thresh = float(args.conf_thresh)

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found. Using default yolov8s.pt model instead.')
    model_path = 'yolov8s.pt'

# Function to create sample database if it doesn't exist
def create_sample_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS inventory (id INTEGER PRIMARY KEY, name TEXT, price INTEGER)")

    csv_file= "inventory.csv"
    with open(csv_file, mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Insert each row into the SQLite table
            cursor.execute("""
            INSERT OR IGNORE INTO inventory (id, name, price)
            VALUES (:id, :name, :price)
            """, row)
        
       
    
    conn.commit()
    conn.close()

create_sample_db()

# Cache database results at the start
def load_inventory_cache(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, id, price FROM inventory")
    inventory_cache = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    conn.close()
    return inventory_cache

inventory_cache = load_inventory_cache(db_path)

# Function to fetch item details from the cache
def get_item_details(classname):
    return inventory_cache.get(classname, (None, None))



# Function to send data to the API endpoint
def send_to_api(payload, api_endpoint):
    try:
        response = requests.post(api_endpoint, json=payload, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f'Sent data to API: {response.status_code}, Response: {response.text}')
    except requests.exceptions.RequestException as e:
        print(f'Failed to send data to API: {e}')

# Load the model
model = YOLO(model_path, task='detect' )
model.export(format="openvino")
ov_model = YOLO('my_model_openvino_model/')
labels = model.names

# Parse input source type
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        source_type = 'image'
    elif ext.lower() in ['.avi', '.mov', '.mp4', '.mkv', '.wmv']:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('http://') or img_source.startswith('https://'):
    source_type = 'stream'
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(img_source + '/*') if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
elif source_type in ['video', 'usb', 'stream']:
    cap_arg = img_source if source_type == 'video' else (usb_idx if source_type == 'usb' else img_source)
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Function to dynamically generate bounding box colors

# def get_color(classidx):
#     np.random.seed(classidx)
#     return tuple(map(int, np.random.randint(0, 255, size=3)))
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]
# Initialize variables
img_count = 0



# Function to process a frame
# def process_frame(frame):
#     results = ov_model(frame, verbose=False)
#     detections = results[0].boxes
#     object_count = 0  

#     deepsort_detections = []
#     filtered_detections = []

#     for i in range(len(detections)):
#         xyxy_tensor = detections[i].xyxy.cpu()
#         xmin, ymin, xmax, ymax = xyxy_tensor.numpy().squeeze().astype(int)

#         classidx = int(detections[i].cls.item())
#         classname = labels[classidx] if classidx < len(labels) else f"Unknown_{classidx}"
#         conf = detections[i].conf.item()
#         # Print debug information
#         print(f"Class Index: {classidx}, Class Name: {classname}, Confidence: {conf}")

#         if conf > conf_thresh:  # Confidence threshold to match
#             item_id, price = get_item_details(classname)
#             if item_id and price:
#                 payload = {
#                     "id": item_id,
#                     "name": classname,
#                     "price": price,
#                     "quantity": 1,
#                     "payable": price
#                 }
#                 send_to_api(payload, api_endpoint)

#             color = bbox_colors[classidx % len(bbox_colors)]
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

#             # Draw the label
#             label = f'{classname}: {int(conf * 100)}%'
#             label_ymin = max(ymin, 10)
#             cv2.rectangle(frame, (xmin, label_ymin - 10), (xmin + 100, label_ymin + 5), color, cv2.FILLED)
#             cv2.putText(frame, label, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

#             object_count += 1
#     cv2.imshow('YOLO detection results', frame)



# Initialize DeepSort tracker
tracker = DeepSort(
    max_age=60,  # Keep tracks alive for 60 frames after disappearance
    n_init=3,  # Require 3 detections to confirm a track
    max_cosine_distance=0.5,  # Increased threshold for re-identification
    nn_budget=100,  # Store 100 embeddings per track
    embedder="mobilenet",  # Use MobileNet for feature extraction
    half=True,  # Use half-precision for faster computation
    bgr=True,  # Input images are in BGR format
    polygon=False,  # Use rectangular bounding boxes
)

# Store processed tracks with timestamps and embeddings
processed_tracks = {}  # track_id: {"logged": bool, "embedding": np.array}

# COOLDOWN_SECONDS = 5  # Prevent duplicate processing

def preprocess_crop(crop, target_size=(224, 224)):
    """Preprocess the cropped image for the embedder."""
    # Resize the crop
    crop = cv2.resize(crop, target_size)

    # Normalize to [0, 1]
    crop = crop.astype(np.float32) / 255.0

    # Convert to [C, H, W] format
    crop = np.transpose(crop, (2, 0, 1))

    # Add batch dimension [1, C, H, W]
    crop = np.expand_dims(crop, axis=0)

    # Convert to PyTorch tensor
    crop = torch.from_numpy(crop).float()

    return crop



# Load a pre-trained MobileNet model as the embedder
embedder = models.mobilenet_v2(pretrained=True)
embedder.classifier = torch.nn.Identity()  # Remove the final classification layer
embedder.eval()  # Set to evaluation mode

def extract_embedding(crop, embedder):
    """Extract embedding from a preprocessed crop."""
    with torch.no_grad():  # Disable gradient calculation
        embedding = embedder(crop)  # Forward pass through the embedder
    return embedding.numpy()  # Convert to NumPy array






def process_frame(frame,embedder):
    frame = cv2.resize(frame, (640, 480))
    results = ov_model(frame, imgsz=480, verbose=False)
    detections = results[0].boxes

    

    # Prepare DeepSort detections
    deepsort_detections = []
    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xmin, ymin, xmax, ymax = xyxy_tensor.numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        global conf
        conf = detections[i].conf.item()

        if conf > conf_thresh:
            width = xmax - xmin
            height = ymax - ymin
            deepsort_detections.append(([xmin, ymin, width, height], conf, classidx))

    # Update tracker
    tracks = tracker.update_tracks(deepsort_detections, frame=frame)
    active_track_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        active_track_ids.add(track_id)
        bbox = track.to_ltrb()
        class_id = track.get_det_class()
        classname = labels[class_id] if class_id < len(labels) else f"Unknown_{class_id}"

        # Initialize track state
        if track_id not in processed_tracks:
            processed_tracks[track_id] = {"logged": False, "embedding": None}

        # Extract embedding for the current track
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:  # Ensure the crop is valid
            crop = preprocess_crop(crop)

            embedding = extract_embedding(crop,embedder)    # Extract embedding

            processed_tracks[track_id]["embedding"] = embedding

        # Check if this is a re-identified track
        for old_track_id, old_track_data in processed_tracks.items():
            if old_track_id == track_id:
                continue  # Skip the current track
            if old_track_data["embedding"] is not None and embedding is not None:
                # Compare embeddings using cosine similarity
                similarity = cosine_similarity(
                    old_track_data["embedding"].reshape(1, -1),
                    embedding.reshape(1, -1))
                if similarity > 0.7:  # Adjust threshold as needed
                    # Re-identified track: Use the old track ID
                    track_id = old_track_id
                    break

        # Process only once per object
        if not processed_tracks[track_id]["logged"]:
            item_id, price = get_item_details(classname)
            if item_id and price:
                payload = {
                    "id": track_id,
                    "name": classname,
                    "price": price,
                    "quantity": 1,
                    "payable": price
                }
                send_to_api(payload, api_endpoint)
                processed_tracks[track_id]["logged"] = True

        # Visualization (always update)
        x1, y1, x2, y2 = map(int, bbox)
        color = bbox_colors[class_id % len(bbox_colors)]
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Draw label (original style)
        label = f'{classname}: {int(conf * 100)}%'  # Use conf from track
        label_ymin = max(y1, 10)
        cv2.rectangle(frame, (x1, label_ymin - 10),
                     (x1 + 100, label_ymin + 5), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, label_ymin),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Cleanup tracks that left the scene
    for track_id in list(processed_tracks.keys()):
        if track_id not in active_track_ids:
            del processed_tracks[track_id]

    cv2.imshow('DeepSort Tracking', frame)
    return frame
# Begin inference loop
while True:
    # Load frame from source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        frame = cv2.imread(imgs_list[img_count])
        if frame is None:
            print(f"Error: Unable to load image {imgs_list[img_count]}")
            break
        img_count += 1

    elif source_type in ['video', 'usb', 'stream']:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file or stream error. Exiting program.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Process the frame
    process_frame(frame,embedder)

    # Wait for key press
    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'stream'] else 0)
    if key == ord('q'):
        break

# Cleanup
if source_type in ['video', 'usb', 'stream']:
    cap.release()
cv2.destroyAllWindows()