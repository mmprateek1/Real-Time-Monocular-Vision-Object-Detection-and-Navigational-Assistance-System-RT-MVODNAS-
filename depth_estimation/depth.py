import cv2
import numpy as np
import torch

class DepthEstimator:
    def __init__(self):
        # Initialize depth estimation model (MiDaS)
        self.model_type = "MiDaS_small"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Load model with GPU optimizations
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Initialize transforms
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform
        
        # Cache for frame size
        self.frame_size = None
        self.target_size = (384, 384)  # Optimal size for MiDaS small
        
        # Initialize CUDA stream for parallel processing
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        
    def estimate(self, frame):
        # Update frame size if changed
        if self.frame_size != frame.shape[:2]:
            self.frame_size = frame.shape[:2]
            # Resize frame for faster processing
            frame = cv2.resize(frame, self.target_size)
        
        # Transform input for model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        # Run model with GPU optimizations
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.cuda.stream(self.stream):
                        prediction = self.midas(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=img.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()
                else:
                    prediction = self.midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
        
        # Convert to numpy and normalize efficiently
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = (depth_map * 255).astype(np.uint8)
        
        # Apply colormap efficiently
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        # Resize back to original size if needed
        if self.frame_size != self.target_size:
            depth_map = cv2.resize(depth_map, (self.frame_size[1], self.frame_size[0]))
        
        return depth_map