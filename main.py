from gui_interface import VisionAssistantGUI
from tools.performance_monitor import PerformanceMonitor
import torch
import cv2
from config import *

def setup_environment():
   
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(True)
    
    
    if torch.cuda.is_available() and USE_CUDA:
        torch.backends.cudnn.benchmark = CUDA_BENCHMARK
        torch.backends.cudnn.enabled = CUDA_ENABLED
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

def main():
    
    setup_environment()
    
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # Start the GUI
        app = VisionAssistantGUI()
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        monitor.stop_monitoring()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()