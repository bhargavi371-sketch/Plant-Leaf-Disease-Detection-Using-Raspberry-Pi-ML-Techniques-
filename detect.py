import cv2
import joblib
import numpy as np
import argparse
import os
from datetime import datetime
from sensor import read_dht11_sensor, read_soil_moisture_sensor

# === Save captured image with timestamp ===
def save_captured_leaf(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_leaf_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"[💾] Saved image as {filename}")

# === Feature Extraction ===
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().reshape(1, -1)

# === Leaf Detection ===
def detect_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > 8000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            if 0.5 < aspect_ratio <2.0:
                return True
    return False

# === Webcam Capture ===
def capture_from_webcam():
    print("[INFO] Capturing from webcam. Press SPACE to capture.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[✘] Failed to access webcam.")
            break
        cv2.imshow("Press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar
            cap.release()
            cv2.destroyAllWindows()
            return cv2.resize(frame, (224, 224))
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()

# === PiCamera Capture ===
def capture_from_picamera():
    try:
        import picamera
        from picamera.array import PiRGBArray
        camera = picamera.PiCamera()
        camera.resolution = (244, 224)
        raw_capture = PiRGBArray(camera) 
        camera.capture(raw_capture, format="bgr")
        image=raw_capture.array
        camera.close()
        print("Image captured from PiCamera")
        return image
    except ImportError:
        print("PiCamera not found,Try --webcam instead.")
        exit()

# === Load Image from Path ===
def load_image_from_file(image_path):
    if not os.path.exists(image_path):
        print(f"[✘] File not found: {image_path}")
        exit()
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Failed to load image from file.")
    print(f"[✔] Loaded image: {image_path}")
    return cv2.resize(image, (224, 224))

# === CLI Arguments ===
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--webcam', action='store_true', help="Capture from webcam")
group.add_argument('--picamera', action='store_true', help="Capture from PiCamera")
group.add_argument('--image', type=str, help="Path to image file")
args = parser.parse_args()

# === Load Model ===
print("[INFO] Model loading...")
model = joblib.load("/home/pi/project/bin/plant_disease_detection_v3/crop_disease_model.pkl")
print("[✔] Model loaded.")

# === Acquire Image ===
if args.webcam:
    image = capture_from_webcam()
elif args.picamera:
    image = capture_from_picamera()
elif args.image:
    image = load_image_from_file(args.image)
    
# === Validate Image===
if image is None or image.size==0:
    print("[x] Failed to capture a valid image. Aborting.")
    exit()
    

# === Leaf Detection & Prediction ===
if detect_leaf(image):
    print("[✔] Leaf detected in the image.")
    save_captured_leaf(image)
    features = extract_features(image)
    prediction = model.predict(features)[0]
    temperature, humidity = read_dht11_sensor()
    moisture = read_soil_moisture_sensor()
    print(f"[✔] Prediction: {prediction}")
    print(f"🌡️ Temperature: {temperature}°C")
    print(f"💧 Humidity: {humidity}%")
    print(f"🌱 Soil Moisture: {moisture}%")
else:
    print("[✘] No leaf detected in the image. Aborting prediction.")
