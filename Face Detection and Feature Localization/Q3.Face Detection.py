#!/usr/bin/python
"""
Advanced Face Detection and Feature Localization - Enhanced Nose Detection
Improved nose detection using multiple classifiers and advanced algorithms
"""

import cv2
import numpy as np
import os

def analyze_computer_vision_problem():
    """Explain the computer vision problem"""
    print("="*70)
    print("COMPUTER VISION PROBLEM ANALYSIS")
    print("="*70)
    print("Problem Type: FACE DETECTION AND FEATURE LOCALIZATION")
    print("\nJustification:")
    print("1. OBJECT DETECTION: Using optimized Haar cascade classifiers")
    print("2. FEATURE EXTRACTION: Enhanced eye, nose, and mouth detection algorithms")
    print("3. GEOMETRIC ANALYSIS: Improved facial feature positioning with multiple methods")
    print("4. MULTI-CLASSIFIER APPROACH: Combining cascade classifiers with geometric estimation")
    print("5. ADAPTIVE PROCESSING: Dynamic parameter adjustment for better accuracy")
    print("="*70)

class AdvancedFaceDetector:
    """Advanced face detection with improved nose detection"""
    
    def __init__(self):
        # Load multiple classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.eye_tree_glasses = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Verify classifiers loaded correctly
        self.check_classifiers()
        
        print("✅ Advanced face detector initialized successfully!")
        
        # Enhanced detection parameters
        self.face_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 6,
            'minSize': (30, 30),
            'maxSize': (800, 800)
        }
        
        self.eye_params = {
            'scaleFactor': 1.1,
            'minNeighbors': 3,
            'minSize': (20, 20)
        }
        
        self.nose_params = {
            'scaleFactor': 1.2,
            'minNeighbors': 4,
            'minSize': (20, 15)
        }
    
    def check_classifiers(self):
        """Verify all classifiers are loaded correctly"""
        classifiers = {
            'Face': self.face_cascade,
            'Eyes': self.eye_cascade,
            'Eyes with Glasses': self.eye_tree_glasses,
            'Nose': self.nose_cascade,
            'Mouth': self.mouth_cascade
        }
        
        for name, classifier in classifiers.items():
            if classifier.empty():
                print(f"⚠️  {name} classifier not available")
            else:
                print(f"✅ {name} classifier loaded")
    
    def preprocess_image_advanced(self, image):
        """Enhanced image preprocessing for better feature detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply bilateral filter for edge-preserving smoothing
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return gray
    
    def detect_faces(self, gray_image):
        """Detect faces with multiple parameter sets for better accuracy"""
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=self.face_params['scaleFactor'],
            minNeighbors=self.face_params['minNeighbors'],
            minSize=self.face_params['minSize'],
            maxSize=self.face_params['maxSize']
        )
        
        # If no faces detected, try alternative parameters
        if len(faces) == 0:
            print("🔄 Trying alternative face detection parameters...")
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(50, 50)
            )
        
        return faces
    
    def detect_eyes_enhanced(self, face_roi_gray, face_roi_color):
        """Enhanced eye detection using multiple classifiers"""
        eyes = []
        
        # Try primary eye classifier
        eyes1 = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=self.eye_params['scaleFactor'],
            minNeighbors=self.eye_params['minNeighbors'],
            minSize=self.eye_params['minSize']
        )
        
        # Try secondary classifier for eyes with glasses
        eyes2 = self.eye_tree_glasses.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(15, 15)
        )
        
        # Combine results, removing duplicates
        all_eyes = list(eyes1) + list(eyes2)
        
        # Remove duplicate detections using IoU
        if len(all_eyes) > 0:
            eyes = self._remove_duplicate_detections(all_eyes)
        
        # Filter eyes based on position and size
        filtered_eyes = []
        face_height, face_width = face_roi_gray.shape
        
        for (ex, ey, ew, eh) in eyes:
            # Eyes should be in upper half of face
            if ey < face_height * 0.6:
                # Reasonable aspect ratio for eyes
                aspect_ratio = ew / eh
                if 0.3 < aspect_ratio < 2.0:
                    # Reasonable size relative to face
                    if ew < face_width * 0.4 and eh < face_height * 0.4:
                        filtered_eyes.append((ex, ey, ew, eh))
        
        # Sort by x-coordinate and return max 2 eyes
        filtered_eyes.sort(key=lambda eye: eye[0])
        return filtered_eyes[:2]
    
    def detect_nose_advanced(self, face_roi_gray, face_bbox, eyes):
        """Advanced nose detection using multiple methods"""
        x, y, w, h = face_bbox
        
        print("   👃 Using multiple nose detection methods...")
        
        # Method 1: Try nose cascade classifier
        nose_detections = []
        if not self.nose_cascade.empty():
            try:
                noses = self.nose_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=self.nose_params['scaleFactor'],
                    minNeighbors=self.nose_params['minNeighbors'],
                    minSize=self.nose_params['minSize']
                )
                nose_detections.extend(noses)
                
                # Try alternative parameters
                noses_alt = self.nose_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(15, 10)
                )
                nose_detections.extend(noses_alt)
            except Exception as e:
                print(f"   ⚠️  Nose classifier error: {e}")
        
        # Remove duplicates
        unique_noses = self._remove_duplicate_detections(nose_detections, threshold=0.3)
        
        # Method 2: Geometric estimation based on eyes
        geometric_nose = self.estimate_nose_geometric(eyes, face_bbox)
        
        # Method 3: Facial proportions-based estimation
        proportional_nose = self.estimate_nose_proportional(face_bbox)
        
        # Choose the best method
        if unique_noses:
            # Use classifier detection if available
            best_nose = max(unique_noses, key=lambda n: n[2] * n[3])
            nx, ny, nw, nh = best_nose
            nose_center = (x + nx + nw//2, y + ny + nh//2)
            method = "Classifier"
            confidence = "High"
        elif len(eyes) == 2:
            # Use geometric estimation if eyes are detected
            nose_center = geometric_nose
            method = "Geometric (with eyes)"
            confidence = "Medium"
        else:
            # Fallback to proportional estimation
            nose_center = proportional_nose
            method = "Proportional"
            confidence = "Low"
        
        print(f"   ✅ Nose detection method: {method} (Confidence: {confidence})")
        return nose_center, method, confidence
    
    def estimate_nose_geometric(self, eyes, face_bbox):
        """Geometric estimation of nose position using eye positions"""
        x, y, w, h = face_bbox
        
        if len(eyes) == 2:
            # Calculate eye centers
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_centers.append((x + ex + ew//2, y + ey + eh//2))
            
            # Calculate midpoint between eyes
            mid_x = (eye_centers[0][0] + eye_centers[1][0]) // 2
            mid_y = (eye_centers[0][1] + eye_centers[1][1]) // 2
            
            # Calculate eye distance for proportional positioning
            eye_distance = abs(eye_centers[1][0] - eye_centers[0][0])
            
            # Refined nose positioning based on facial proportions
            # Nose is typically at 0.6-0.7x eye distance below eye line
            nose_y = mid_y + int(eye_distance * 0.65)
            nose_x = mid_x
            
            # Ensure nose is within reasonable face bounds
            nose_y = min(max(nose_y, y + h//3), y + 2*h//3)
            
        else:
            # Fallback to proportional estimation
            nose_x, nose_y = self.estimate_nose_proportional(face_bbox)
        
        return (nose_x, nose_y)
    
    def estimate_nose_proportional(self, face_bbox):
        """Estimate nose position based on facial proportions"""
        x, y, w, h = face_bbox
        
        # Standard facial proportions: nose is typically at 2/3 from top to bottom
        nose_x = x + w // 2
        nose_y = y + 2 * h // 3
        
        # Add small random variation to avoid identical positions for different faces
        variation_x = w // 20
        variation_y = h // 15
        
        nose_x += np.random.randint(-variation_x, variation_x)
        nose_y += np.random.randint(-variation_y, variation_y)
        
        return (nose_x, nose_y)
    
    def detect_mouth_enhanced(self, face_roi_gray, face_bbox):
        """Enhanced mouth detection for better facial feature completeness"""
        x, y, w, h = face_bbox
        mouth_detections = []
        
        if not self.mouth_cascade.empty():
            # Mouth should be in lower half of face
            mouth_roi_y_start = int(h * 0.6)
            mouth_roi = face_roi_gray[mouth_roi_y_start:h, 0:w]
            
            if mouth_roi.size > 0:
                try:
                    mouths = self.mouth_cascade.detectMultiScale(
                        mouth_roi,
                        scaleFactor=1.5,
                        minNeighbors=15,
                        minSize=(30, 20)
                    )
                    
                    for (mx, my, mw, mh) in mouths:
                        # Adjust coordinates to full face ROI
                        mx_adj = mx
                        my_adj = my + mouth_roi_y_start
                        mouth_detections.append((mx_adj, my_adj, mw, mh))
                except Exception as e:
                    print(f"   ⚠️  Mouth detection error: {e}")
        
        if mouth_detections:
            # Return the largest mouth detection
            mouth_detections.sort(key=lambda m: m[2] * m[3], reverse=True)
            mx, my, mw, mh = mouth_detections[0]
            mouth_center = (x + mx + mw//2, y + my + mh//2)
            return mouth_center, "Detected"
        else:
            # Estimate mouth position based on nose and face proportions
            mouth_x = x + w // 2
            mouth_y = y + 3 * h // 4
            return (mouth_x, mouth_y), "Estimated"
    
    def _remove_duplicate_detections(self, detections, threshold=0.5):
        """Remove duplicate detections using IoU"""
        if len(detections) <= 1:
            return detections
        
        keep = []
        suppressed = set()
        
        for i in range(len(detections)):
            if i in suppressed:
                continue
            
            keep.append(i)
            rect_i = detections[i]
            
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue
                
                rect_j = detections[j]
                iou = self._calculate_iou(rect_i, rect_j)
                
                if iou > threshold:
                    suppressed.add(j)
        
        return [detections[i] for i in keep]
    
    def _calculate_iou(self, rect1, rect2):
        """Calculate Intersection over Union for two rectangles"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def detect_facial_features_advanced(self, image_path):
        """Advanced facial feature detection with improved nose recognition"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print("🔄 Advanced image preprocessing...")
        # Preprocess image
        gray = self.preprocess_image_advanced(img)
        
        # Detect faces
        print("🔍 Detecting faces...")
        faces = self.detect_faces(gray)
        
        print(f"✅ Detected {len(faces)} face(s)")
        
        if len(faces) == 0:
            print("❌ No faces detected. Trying with alternative preprocessing...")
            # Try without advanced preprocessing
            gray_simple = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_simple, 1.3, 5)
            if len(faces) == 0:
                return None
        
        # Create annotated image
        annotated_img = img.copy()
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            print(f"\n👤 Processing Face {i+1}:")
            print(f"   Position: ({x}, {y}), Size: {w}x{h}")
            
            # Draw face rectangle
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region for detailed analysis
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = img[y:y+h, x:x+w]
            
            if face_roi_gray.size == 0:
                continue
            
            # Detect eyes with enhanced method
            print("   👀 Detecting eyes...")
            eyes = self.detect_eyes_enhanced(face_roi_gray, face_roi_color)
            eye_centers = []
            
            for j, (ex, ey, ew, eh) in enumerate(eyes):
                if j >= 2:  # Only process first 2 eyes
                    break
                    
                # Calculate eye center in original image coordinates
                center_x = x + ex + ew//2
                center_y = y + ey + eh//2
                eye_centers.append((center_x, center_y))
                
                # Draw eye bounding box and center
                cv2.rectangle(annotated_img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)
                cv2.circle(annotated_img, (center_x, center_y), 4, (255, 255, 0), -1)
                
                print(f"   👁️ Eye {j+1}: Center at ({center_x}, {center_y})")
            
            # Advanced nose detection
            nose_tip, nose_method, nose_confidence = self.detect_nose_advanced(face_roi_gray, (x, y, w, h), eyes)
            
            # Enhanced mouth detection
            mouth_center, mouth_method = self.detect_mouth_enhanced(face_roi_gray, (x, y, w, h))
            
            # Draw features with different colors based on detection method
            self._draw_advanced_annotations(annotated_img, x, y, w, h, eye_centers, 
                                          nose_tip, mouth_center, nose_method, i+1)
            
            results.append({
                'face_id': i+1,
                'bbox': (x, y, w, h),
                'eyes': eye_centers,
                'nose': nose_tip,
                'mouth': mouth_center,
                'nose_method': nose_method,
                'nose_confidence': nose_confidence,
                'mouth_method': mouth_method,
                'eyes_detected': len(eyes)
            })
            
            print(f"   👃 Nose tip: {nose_tip} ({nose_method}, {nose_confidence})")
            print(f"   👄 Mouth: {mouth_center} ({mouth_method})")
        
        return {
            'original': img,
            'annotated': annotated_img,
            'faces': results,
            'total_faces': len(faces)
        }
    
    def _draw_advanced_annotations(self, image, x, y, w, h, eyes, nose, mouth, nose_method, face_id):
        """Draw advanced annotations with method-based coloring"""
        # Colors for different detection methods
        colors = {
            'Classifier': (0, 0, 255),    # Red for classifier-based
            'Geometric': (255, 0, 0),     # Blue for geometric
            'Proportional': (0, 255, 255) # Yellow for proportional
        }
        
        nose_color = colors.get(nose_method, (0, 0, 255))
        
        # Draw nose with method-based color
        cv2.circle(image, nose, 8, nose_color, -1)
        cv2.circle(image, nose, 10, nose_color, 2)
        
        # Draw mouth
        cv2.circle(image, mouth, 6, (255, 0, 255), -1)
        cv2.circle(image, mouth, 8, (255, 0, 255), 2)
        
        # Draw eyes
        for i, (eye_x, eye_y) in enumerate(eyes):
            cv2.circle(image, (eye_x, eye_y), 6, (255, 255, 0), -1)
            cv2.circle(image, (eye_x, eye_y), 8, (255, 255, 0), 2)
        
        # Add labels
        cv2.putText(image, f"Face {face_id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Nose ({nose_method})", (nose[0]-30, nose[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, nose_color, 1)
        cv2.putText(image, "Mouth", (mouth[0]-15, mouth[1]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Draw facial landmarks connections
        if len(eyes) == 2:
            cv2.line(image, eyes[0], eyes[1], (255, 255, 0), 1)
            if nose:
                cv2.line(image, eyes[0], nose, (255, 255, 0), 1)
                cv2.line(image, eyes[1], nose, (255, 255, 0), 1)
            if mouth:
                cv2.line(image, nose, mouth, (255, 255, 0), 1)

def main():
    """Main function"""
    print("🎯 ADVANCED Face Detection and Feature Localization")
    print("==================================================")
    
    # Analyze the computer vision problem
    analyze_computer_vision_problem()
    
    # Get image path
    image_path = input("\n📁 Enter the path to your image: ").strip().replace('"', '')
    
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        print("💡 Try using a full path like: C:/Users/Name/Pictures/face.jpg")
        return
    
    # Initialize advanced detector
    detector = AdvancedFaceDetector()
    
    # Process image
    print(f"\n🔄 Processing image: {os.path.basename(image_path)}")
    result = detector.detect_facial_features_advanced(image_path)
    
    if result is not None:
        # Save result
        output_path = "advanced_detection_result.jpg"
        cv2.imwrite(output_path, result['annotated'])
        print(f"\n💾 Results saved as: {output_path}")
        
        # Display results
        cv2.imshow('Advanced Face Detection - Press any key to close', result['annotated'])
        print("👀 Displaying results... Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Generate detailed report
        print(f"\n📊 ADVANCED DETECTION REPORT:")
        print(f"Total faces detected: {result['total_faces']}")
        
        for face in result['faces']:
            print(f"\nFace {face['face_id']}:")
            print(f"  Bounding box: {face['bbox']}")
            print(f"  Eyes detected: {face['eyes_detected']} at {face['eyes']}")
            print(f"  Nose: {face['nose']} (Method: {face['nose_method']}, Confidence: {face['nose_confidence']})")
            print(f"  Mouth: {face['mouth']} (Method: {face['mouth_method']})")
        
        print(f"\n✅ Advanced processing completed successfully!")
        
    else:
        print("❌ No faces detected or error occurred.")
        print("💡 Tips for better detection:")
        print("   - Use high-quality, well-lit images")
        print("   - Ensure faces are clearly visible and front-facing")
        print("   - Try different angles or expressions")

if __name__ == "__main__":
    main()
