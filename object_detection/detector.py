import cv2
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        try:
            # Initialize YOLO with error handling
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO('yolov8n.pt')
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Enable CUDA optimizations if available
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
            
            self.class_names = None
            self.frame_size = None
            
            # Set confidence thresholds
            self.conf_threshold = 0.25
            self.iou_threshold = 0.45
            
            print("ObjectDetector initialized successfully")
        except Exception as e:
            print(f"Error initializing ObjectDetector: {str(e)}")
            raise
    
    def detect(self, frame):
        try:
            # Ensure frame is valid and convert to RGB if needed
            if frame is None:
                return [], frame
            
            # Update frame size and resize if needed
            if self.frame_size != frame.shape[:2]:
                self.frame_size = frame.shape[:2]
                
            # Ensure frame is contiguous in memory
            frame = np.ascontiguousarray(frame)
            
            # Run inference
            results = self.model(frame, 
                               conf=self.conf_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            # Get class names if not already cached
            if self.class_names is None:
                self.class_names = results[0].names
            
            detected_objects = []
            
            # Process results
            if len(results) > 0:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {confidence:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detected_objects, frame
            
        except Exception as e:
            print(f"Error in detect: {str(e)}")
            return [], frame