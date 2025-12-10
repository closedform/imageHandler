from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import cv2
from PIL import Image

class DepthEstimator:
    def __init__(self, model_id="LiheYoung/depth-anything-small-hf"):
        print(f"Loading depth model: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # load model
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except Exception:
             # Fallback if fast processor not available
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def predict(self, frame):
        # Frame is BGR numpy array
        # Resize for speed? The model has its own sizing, but sending huge images is slow.
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (518, 518)) # Depth Anything standard training size is 518
        
        image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        return depth_map
