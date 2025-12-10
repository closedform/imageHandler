import sys
import os

try:
    print("Testing imports...")
    import cv2
    import mediapipe
    import open3d
    import torch
    import transformers
    import numpy
    print("Imports success.")
    
    print("Testing Model Download/Load...")
    from src.depth import DepthEstimator
    # This might take a while if downloading
    d = DepthEstimator()
    print("Depth Model loaded successfully.")
    
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("Verification passed.")
