import cv2
import numpy as np
import torch
from object_detection.detector import ObjectDetector
from depth_estimation.depth import DepthEstimator
from navigation.navigator import Navigator
from env_description.descriptor import EnvironmentDescriptor
from face_analysis.analyzer import FaceAnalyzer

def test_module(module_name, module_instance, frame):
    print(f"\nTesting {module_name}...")
    try:
        if module_name == "Object Detector":
            objects, annotated_frame = module_instance.detect(frame)
            cv2.imshow('Object Detection', annotated_frame)
            print(f"Detected {len(objects)} objects")
            
        elif module_name == "Depth Estimator":
            depth_map = module_instance.estimate(frame)
            cv2.imshow('Depth Map', depth_map)
            print("Depth map generated")
            
        elif module_name == "Navigator":
            # Need both objects and depth map for navigation
            detector = ObjectDetector()
            depth_estimator = DepthEstimator()
            objects, _ = detector.detect(frame)
            depth_map = depth_estimator.estimate(frame)
            guidance = module_instance.get_guidance(objects, depth_map)
            print("Navigation guidance:", guidance['recommendation'])
            
        elif module_name == "Environment Descriptor":
            # Need objects and depth map for environment description
            detector = ObjectDetector()
            depth_estimator = DepthEstimator()
            objects, _ = detector.detect(frame)
            depth_map = depth_estimator.estimate(frame)
            env_info = module_instance.analyze(frame, objects, depth_map)
            print("Environment description:", env_info['description'])
            
        elif module_name == "Face Analyzer":
            faces = module_instance.analyze(frame)
            print(f"Detected {len(faces)} faces")
            
        print(f"{module_name} test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error testing {module_name}: {str(e)}")
        return False

def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get a test frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    # Initialize modules
    modules = {
        "Object Detector": ObjectDetector(),
        "Depth Estimator": DepthEstimator(),
        "Navigator": Navigator(),
        "Environment Descriptor": EnvironmentDescriptor(),
        "Face Analyzer": FaceAnalyzer()
    }
    
    # Test each module
    results = {}
    for name, module in modules.items():
        results[name] = test_module(name, module, frame.copy())
        
        # Wait for key press between tests
        cv2.waitKey(0)
    
    # Print summary
    print("\nTest Summary:")
    for name, success in results.items():
        print(f"{name}: {'PASS' if success else 'FAIL'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()