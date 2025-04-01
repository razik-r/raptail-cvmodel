import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
import torch

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

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found. Using default yolov8s.pt model instead.')
    model_path = 'yolov8s.pt'

# Load the model
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input source type
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
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

# Check if recording is valid and set up recording

# if record:
#     if source_type not in ['video', 'usb', 'stream']:
#         print('Recording only works for video, camera, and stream sources. Please try again.')
#         sys.exit(0)
#     if not user_res:
#         print('Please specify resolution to record video at.')
#         sys.exit(0)
    
#     # Set up recording
#     record_name = 'demo1.avi'
#     record_fps = 30
#     recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [file for file in glob.glob(img_source + '/*') if os.path.splitext(file)[1] in img_ext_list]
elif source_type in ['video', 'usb', 'stream']:
    cap_arg = img_source if source_type == 'video' else (usb_idx if source_type == 'usb' else img_source)
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set bounding box colors
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]

# Initialize variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    elif source_type in ['video', 'usb', 'stream']:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file or stream error. Exiting program.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference on frame
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xmin, ymin, xmax, ymax = xyxy_tensor.numpy().squeeze().astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf * 100)}%'
            label_ymin = max(ymin, 10)
            cv2.rectangle(frame, (xmin, label_ymin - 10), (xmin + 100, label_ymin + 5), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

    # Display FPS and object count
    if source_type in ['video', 'usb', 'stream']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.imshow('YOLO detection results', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video', 'usb', 'stream'] else 0)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_buffer.append(1 / (t_stop - t_start))
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
if source_type in ['video', 'usb', 'stream']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
