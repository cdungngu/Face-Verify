import cv2
import numpy as np
from insightface.app import FaceAnalysis
from dataclasses import dataclass
import os
import time

# Lấy thư mục chứa file hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Constants and configuration
@dataclass
class IDCardSpecs:
    aspect_ratio: float
    template_features: dict

    @staticmethod
    def create_standard_id():
        return IDCardSpecs(
            aspect_ratio=1.58,  # Standard ID card aspect ratio
            template_features={
                "logo": (0.05, 0.05, 0.3, 0.2),     # x1, y1, x2, y2 in percentage
                "photo_area": (0.7, 0.2, 0.95, 0.7),  # x1, y1, x2, y2 in percentage
            }
        )

# Face detection functions
def init_face_detector():
    """Initialize face detector with option for lightweight detection"""
    # Giả sử các file model nằm trong thư mục "models" trong cùng thư mục với file này
    model_file = os.path.join(current_dir, "models", "opencv_face_detector_uint8.pb")
    config_file = os.path.join(current_dir, "models", "opencv_face_detector.pbtxt")
    
    # Check if files exist, if not, download or provide instructions
    # You'll need to download these files or use ones from OpenCV's face_detector model
    
    detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    return {'type': 'opencv_dnn', 'model': detector}


def detect_faces(detector, image, min_confidence=0.92, min_face_size=20):
    """Detect faces in an image with performance optimizations"""
    # Resize image for faster processing (optional)
    scale_factor = 1.0  # Reduce to 0.5 for even faster processing
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    
    if detector['type'] == 'opencv_dnn':
        # OpenCV DNN-based detection (much faster)
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        detector['model'].setInput(blob)
        detections = detector['model'].forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Convert to MTCNN-compatible format
                faces.append({
                    'box': [x1, y1, x2-x1, y2-y1],
                    'confidence': float(confidence),
                    'keypoints': {}  # No keypoints with this detector
                })
        
        return faces
    else:
        # Original MTCNN implementation
        return detector['model'].detect_faces(image, min_face_size=min_face_size)

def draw_face(image, faces):
    """Draw rectangle around detected face"""
    if not faces:
        return image
    
    display_image = image.copy()
    x, y, w, h = faces[0]['box']
    confidence = faces[0]['confidence']
    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{w} x {h}, conf: {confidence:.2f}"
    cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return display_image

# Face comparison functions
def init_face_analysis(det_size=(640, 640)):
    """Initialize face analysis"""
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def get_face_embedding(app, image):
    """Extract face embedding from image"""
    faces = app.get(image)
    if len(faces) == 0:
        return None
    face = faces[0]
    if hasattr(face, 'det_score') and face.det_score < 0.5:
        print(f"Warning: Low quality face detection (score: {face.det_score:.2f})")
    
    return face.embedding

def compare_face_embeddings(feat1, feat2):
    """Compare two face embeddings and return similarity score"""
    if feat1 is None or feat2 is None:
        return -1.0  # Return negative value to indicate invalid comparison
        
    # Normalize vectors to unit length
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    
    # Calculate cosine similarity
    sim = np.dot(feat1, feat2)
    return sim

def is_same_person(feat1, feat2, threshold=0.3):
    """Check if two face embeddings belong to the same person"""
    if feat1 is None or feat2 is None:
        return False
    return compare_face_embeddings(feat1, feat2) > threshold

# ID card detection functions
def extract_reference_features(reference_image, card_specs):
    """Extract features from reference ID card image"""
    # Convert to grayscale for feature extraction
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Store reference aspect ratio and features
    h, w = reference_image.shape[:2]
    reference_features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "aspect_ratio": float(w) / h,
        "width": w,
        "height": h
    }
    
    return reference_features

def detect_id_card(frame, reference, reference_features, card_specs, min_matches=15):
    """
    Detect ID card in frame using ORB feature matching with stability improvements
    """
    # Convert frame to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve feature detection
    gray = cv2.equalizeHist(gray)
    
    # Initialize ORB detector with more features and better parameters
    orb = cv2.ORB_create(
        nfeatures=2000,          # Increase number of features
        scaleFactor=1.2,         # Smaller scale factor for better multi-scale detection
        nlevels=8,               # More scale levels
        edgeThreshold=31,        # Avoid features at image borders
        firstLevel=0,
        WTA_K=2,
        patchSize=31,            # Larger patch size for more distinctive features
        fastThreshold=20         # Adjust FAST detector threshold
    )
    
    # Detect keypoints and compute descriptors for current frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray, None)
    
    # If no features found, return early
    if descriptors_frame is None or len(keypoints_frame) < 8:
        return False, None, None
    
    # Create feature matcher - use KNN matcher instead of BFMatcher with crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors between reference and current frame using KNN
    matches = bf.knnMatch(reference_features["descriptors"], descriptors_frame, k=2)
    
    # Apply ratio test to filter good matches (Lowe's ratio test)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < min_matches:
        return False, None, None
    
    ref_pts = np.float32([reference_features["keypoints"][m.queryIdx].pt for m in good_matches])
    frame_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches])
    
    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        return False, None, None
    
    inlier_count = np.sum(mask)
    
    if inlier_count < min_matches * 0.7:
        return False, None, None
    
    h, w = reference.shape[:2]
    ref_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    frame_corners = cv2.perspectiveTransform(ref_corners, H)
    corners = np.int32(frame_corners)
    
    if not cv2.isContourConvex(corners):
        return False, None, None
    
    detected_width = max(
        np.linalg.norm(corners[0][0] - corners[3][0]),
        np.linalg.norm(corners[1][0] - corners[2][0])
    )
    detected_height = max(
        np.linalg.norm(corners[0][0] - corners[1][0]),
        np.linalg.norm(corners[2][0] - corners[3][0])
    )
    
    if detected_height < 50 or detected_width < 50 or detected_height == 0:
        return False, None, None
        
    detected_aspect = detected_width / detected_height
    aspect_tolerance = 0.2
    if abs(detected_aspect - card_specs.aspect_ratio) > aspect_tolerance * card_specs.aspect_ratio:
        return False, None, None
    
    frame_height, frame_width = frame.shape[:2]
    total_area = frame_width * frame_height
    card_area = cv2.contourArea(corners)
    
    min_area_percentage = 0.02
    max_area_percentage = 0.9
    if card_area / total_area < min_area_percentage or card_area / total_area > max_area_percentage:
        return False, None, None
    
    target_w = 600
    target_h = int(target_w / card_specs.aspect_ratio)
    
    dst_pts = np.float32([
        [0, 0],
        [0, target_h - 1],
        [target_w - 1, target_h - 1],
        [target_w - 1, 0]
    ])
    
    M = cv2.getPerspectiveTransform(frame_corners.reshape(4, 2).astype(np.float32), dst_pts)
    warped = cv2.warpPerspective(frame, M, (target_w, target_h))
    
    return True, corners, warped

def draw_card(frame, corners):
    """Draw detected card boundaries"""
    if corners is None:
        return frame
        
    display_frame = frame.copy()
    cv2.drawContours(display_frame, [corners], -1, (0, 255, 0), 3)
    return display_frame

def main():
    # Các biến ban đầu trong main()
    stable_frames_required = 30
    stable_frame_counter = 0
    last_detection_state = False
    
    # Khởi tạo face detector và face analyzer
    face_detector = init_face_detector()
    face_analyzer = init_face_analysis()
    
    # Khởi tạo ID card detector với ảnh tham chiếu
    # Xây dựng đường dẫn tuyệt đối cho reference_id_card.jpg
    reference_path = os.path.join(current_dir, "reference_id_card.jpg")
    reference = cv2.imread(reference_path)
    if reference is None:
        print("Could not load reference image")
        return
        
    card_specs = IDCardSpecs.create_standard_id()
    reference_features = extract_reference_features(reference, card_specs)
    
    cap = cv2.VideoCapture(0) # chú ý ở chỗ này
    state = "DETECT_CARD"
    captured_card = None
    card_face_embedding = None
    live_face_embedding = None
    
    print("=== Face Verification System ===")
    print("Step 1: Detecting ID card automatically")
    
    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
            
        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('r') and state == "SHOW_RESULT":
            state = "DETECT_CARD"
            captured_card = None
            card_face_embedding = None
            live_face_embedding = None
            stable_frame_counter = 0
            print("Restarting verification process")
            print("Step 1: Detecting ID card automatically")
        
        if state == "DETECT_CARD":
            card_detected, corners, warped = detect_id_card(frame, reference, reference_features, card_specs)
            
            if card_detected:
                display = draw_card(display, corners)
                cv2.putText(display, "ID Card Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                stable_frame_counter = stable_frame_counter + 1 if last_detection_state else 0
                
                if stable_frame_counter >= stable_frames_required:
                    coords = card_specs.template_features["photo_area"]
                    x1, y1, x2, y2 = [int(c * d) for c, d in zip(coords, [warped.shape[1], 
                                                                            warped.shape[0], 
                                                                            warped.shape[1], 
                                                                            warped.shape[0]])]
                    card_face_img = warped[y1:y2, x1:x2]
                    card_face_embedding = get_face_embedding(face_analyzer, card_face_img)
                    
                    if card_face_embedding is not None:
                        captured_card = warped.copy()
                        state = "DETECT_FACE"
                        stable_frame_counter = 0
                        print("Step 2: Detecting live face automatically")
                    else:
                        stable_frame_counter = 0
                        cv2.putText(display, "No face found on card, try again", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Move ID card into view", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                stable_frame_counter = 0
            
            last_detection_state = card_detected
            
        elif state == "DETECT_FACE":
            faces = detect_faces(face_detector, frame)
            
            if faces:
                display = draw_face(display, faces)
                cv2.putText(display, "Face Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                stable_frame_counter = stable_frame_counter + 1 if last_detection_state else 0
                
                if stable_frame_counter >= stable_frames_required:
                    live_face_embedding = get_face_embedding(face_analyzer, frame)
                    
                    if live_face_embedding is not None:
                        state = "SHOW_RESULT"
                        stable_frame_counter = 0
                        print("Step 3: Comparing faces...")
                    else:
                        stable_frame_counter = 0
                        cv2.putText(display, "Face analysis failed, try again", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Position your face in the camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                stable_frame_counter = 0
                
            last_detection_state = len(faces) > 0
                
        elif state == "SHOW_RESULT":
            if card_face_embedding is not None and live_face_embedding is not None:
                similarity = compare_face_embeddings(card_face_embedding, live_face_embedding)
                is_match = is_same_person(card_face_embedding, live_face_embedding)
                
                confidence_score = min(100, max(0, int((similarity - 0.2) * 200)))
                
                result_color = (0, 255, 0) if is_match else (0, 0, 255)
                result_text = "MATCH VERIFIED" if is_match else "NO MATCH"
                
                cv2.putText(display, f"Result: {result_text}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                cv2.putText(display, f"Similarity: {similarity:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, "Press 'r' to restart, 'q' to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display, "Error: Failed to extract face features", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, "Press 'r' to restart, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Verification System', display)
    
    print("Exiting face verification system")
    cap.release()
    cv2.destroyAllWindows(0)

if __name__ == "__main__":
    main()
