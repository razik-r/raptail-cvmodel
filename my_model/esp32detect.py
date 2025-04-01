import cv2
import numpy as np
import urllib.request

# Replace with your ESP32-CAM stream URL
url = 'http://192.168.1.9:81/stream'

# Open the stream
stream = urllib.request.urlopen(url)
bytes = bytes()

while True:
    bytes += stream.read(1024)  # Read 1024 bytes at a time
    a = bytes.find(b'\xff\xd8')  # JPEG start marker
    b = bytes.find(b'\xff\xd9')  # JPEG end marker
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]  # Extract JPEG frame
        bytes = bytes[b+2:]  # Remove processed frame
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('ESP32-CAM Stream', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()