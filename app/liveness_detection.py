from flask import Flask, request, jsonify
import base64
import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import traceback  # Import traceback for error logging
from app import app

#app = Flask(__name__)

#app.config['MAX_CONTENT_LENGTH'] = 5000 * 1024 * 1024
#app.config['MAX_FORM_MEMORY_SIZE'] = 50 * 1024 * 1024  # 50 MB

# Error handler for oversized requests
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Request entity too large", "message": str(error)}), 413

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        # Check if the image field exists in the request
        img_data = request.form.get('image', '')
        if not img_data:
            print("No 'image' field in the request.")
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Log the received base64 size
        print(f"Received base64 image data size: {len(img_data)} bytes")

        # Decode base64 data
        try:
            img_data = base64.b64decode(img_data)
        except Exception as e:
            print(f"Base64 decoding error: {e}")
            return jsonify({"status": "error", "message": "Failed to decode base64 image"}), 400

        print(f"Decoded binary image size: {len(img_data)} bytes")

        # Convert binary data to an OpenCV image
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Decoded image is None. Possible invalid image data.")
            return jsonify({"status": "error", "message": "Failed to decode image"}), 400

        print(f"cv2 image data size: {img.shape if img is not None else 'None'}")
        # Use OpenCV's face detection (for demonstration)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected"}), 400

        if not detect_liveness(img):
                return jsonify({"status": "error", "message": "Fake face detected"}), 400

        for (x, y, w, h) in faces:
            # Convert OpenCV rectangle to dlib rectangle
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            # Get facial landmarks
            landmarks = shape_predictor(gray, dlib_rect)

            # Calculate head pose
            yaw, pitch, roll = calculate_head_pose(landmarks)
            print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

            # Detect head movement
            if detect_head_movement(yaw, pitch, roll):
                return jsonify({"status": "success", "message": "Head movement detected"})

        return jsonify({"status": "error", "message": "No significant head movement detected"}), 400
    
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Respond with detection results
        #return jsonify({"status": "success", "faces_detected": len(faces)})
    
    except Exception as e:
        # Log the error and the stack trace
        print(f"Error: {str(e)}")
        print("Stack trace:")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": "An error occurred during processing"}), 500


# Load pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    # Compute distances between the vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the distance between horizontal eye landmarks
    C = distance.euclidean(eye[0], eye[3])

    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def detect_liveness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    for face in faces:
        shape = shape_predictor(gray, face)
        # Extract eye coordinates
        left_eye = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
        right_eye = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0

        # Threshold for blink detection
        if avg_ear < 0.25:  # Tweak this value based on testing
            return True  # Live face detected via blink

    return False
    
# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype="double")

# Camera matrix (assume focal length = 1 and principal point at center)
focal_length = 1.0
camera_matrix = np.array([
    [focal_length, 0, 0.5],
    [0, focal_length, 0.5],
    [0, 0, 1]
], dtype="double")

# Head movement thresholds
MOVEMENT_THRESHOLD = 15  # Degrees

# Global variables to track head pose
last_yaw, last_pitch, last_roll = None, None, None

def calculate_head_pose(landmarks):
    # 2D image points from detected landmarks
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),    # Chin
        (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
    ], dtype="double")

    # Solve PnP to find rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, None
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Decompose rotation matrix to Euler angles (yaw, pitch, roll)
    yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * (180.0 / np.pi)
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * (180.0 / np.pi)
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * (180.0 / np.pi)

    return yaw, pitch, roll

def detect_head_movement(yaw, pitch, roll):
    global last_yaw, last_pitch, last_roll
    if last_yaw is None or last_pitch is None or last_roll is None:
        # Initialize the previous values
        last_yaw, last_pitch, last_roll = yaw, pitch, roll
        return False

    # Calculate the difference in angles
    delta_yaw = abs(yaw - last_yaw)
    delta_pitch = abs(pitch - last_pitch)
    delta_roll = abs(roll - last_roll)

    # Update the previous values
    last_yaw, last_pitch, last_roll = yaw, pitch, roll

    # Check if movement exceeds the threshold
    return delta_yaw > MOVEMENT_THRESHOLD or delta_pitch > MOVEMENT_THRESHOLD or delta_roll > MOVEMENT_THRESHOLD


'''
@app.route('/detect-liveness', methods=['POST'])
def detect_live():
    try:
        # Get the image from the request
        img_data = request.form.get('image', '')
        if not img_data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        # Decode base64 image
        img_bytes = base64.b64decode(img_data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Preprocess the image
        input_tensor = preprocess_image(frame)

         # Run inference
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()

        # Determine liveness
        is_live = prediction > 0.5

        return jsonify({"status": "success", "is_live": is_live, "confidence": prediction})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

'''
