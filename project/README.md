# Smart Checkout System

A real-time object detection and self-checkout system using computer vision.

## Components

1. **Object Detection (Python)**
   - Simulated computer vision system
   - Detects products in real-time
   - Sends detections to backend API

2. **Backend Server (Node.js)**
   - Express.js REST API
   - Real-time updates with Socket.IO
   - Manages shopping cart and checkout

3. **Frontend (React)**
   - Real-time cart updates
   - Beautiful UI with Tailwind CSS
   - Checkout functionality

## Setup

1. Install dependencies:
   ```bash
   npm install
   pip install requests pillow numpy opencv-python
   ```

2. Start the backend server:
   ```bash
   npm run dev:backend
   ```

3. Start the frontend development server:
   ```bash
   npm run dev:frontend
   ```

4. Run the object detection script:
   ```bash
   python object_detection.py
   ```

## Features

- Real-time object detection
- Automatic product recognition
- Live cart updates
- Beautiful checkout interface
- Mock payment processing

## Note

This is a demonstration system. In a production environment, you would:

1. Use a proper ML model (e.g., YOLOv8)
2. Implement secure payment processing
3. Add user authentication
4. Use a proper database
5. Add error handling and retry mechanisms