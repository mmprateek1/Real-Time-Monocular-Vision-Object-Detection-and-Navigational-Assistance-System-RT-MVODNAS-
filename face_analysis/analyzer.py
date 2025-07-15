import cv2
import numpy as np

class FaceAnalyzer:
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Cache for frame size
        self.frame_size = None
        
        # Optimize detection parameters
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)
        
    def analyze(self, frame):
        # Update frame size if changed
        if self.frame_size != frame.shape[:2]:
            self.frame_size = frame.shape[:2]
            # Resize frame for faster processing if too large
            if frame.shape[0] > 640 or frame.shape[1] > 640:
                scale = min(640/frame.shape[0], 640/frame.shape[1])
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Convert to grayscale efficiently
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        face_data = []
        for (x, y, w, h) in faces:
            face_info = {
                'bbox': (x, y, w, h),
                'confidence': None,
                'position': self._get_face_position(frame, x, y)
            }
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add face region
            face_region = frame[y:y+h, x:x+w]
            if face_region.size > 0:
                # Analyze face characteristics
                face_info.update(self._analyze_face_region(face_region))
            
            face_data.append(face_info)
        
        return face_data
    
    def _get_face_position(self, frame, x, y):
        height, width = frame.shape[:2]
        center_x = x + width//2
        
        # Determine horizontal position
        if center_x < width//3:
            h_pos = "left"
        elif center_x < 2*width//3:
            h_pos = "center"
        else:
            h_pos = "right"
            
        # Determine vertical position
        if y < height//3:
            v_pos = "top"
        elif y < 2*height//3:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        return f"{v_pos}-{h_pos}"
    
    def _analyze_face_region(self, face_region):
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness (rough estimate of lighting conditions)
        brightness = np.mean(gray_face)
        
        # Calculate contrast
        contrast = np.std(gray_face)
        
        return {
            'lighting': 'good' if 85 < brightness < 170 else 'poor',
            'contrast': 'good' if contrast > 40 else 'poor',
            'size': face_region.shape
        }