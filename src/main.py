import cv2
import time
import argparse
from camera import Camera
from hands import HandTracker
from depth import DepthEstimator
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Depth Cloud Hand Control")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--skip", type=int, default=1, help="Skip N frames for depth estimation")
    args = parser.parse_args()

    # Initialize components
    print("Initializing Camera...")
    cam = Camera(device_id=args.camera)
    cam.start()

    print("Initializing Hand Tracker...")
    hands = HandTracker()

    print("Initializing Depth Estimator...")
    # depth_est = DepthEstimator(model_id="depth-anything/Depth-Anything-V2-Small-hf")
    # Using the standard V2 small if available, or V1 small
    # Note: HuggingFace hub path might vary. 
    # Let's stick to the one I put in depth.py, or user can change it.
    depth_est = DepthEstimator() 

    print("Initializing Visualizer...")
    vis = Visualizer(width=cam.width, height=cam.height)
    
    # Background Subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    use_bg_sub = False
    
    print("Starting loop...")
    frame_count = 0
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue
            
            # Update Motion Mask
            fgmask = fgbg.apply(frame)
            motion_mask = None
            
            if use_bg_sub:
                # Cleanup noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                motion_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                # Threshold to binary (0 or 255)
                _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
            
            # Copy for visualization/processing
            display_frame = frame.copy()

            # 1. Hand Tracking
            hand_state = hands.process(display_frame)
            hands.draw_debug(display_frame, hand_state.get("landmarks"))
            
            # 2. Depth Estimation (every N frames)
            # We reuse the last depth map if skipping to maintain speed
            if frame_count % args.skip == 0:
                depth_map = depth_est.predict(frame)
                current_depth = depth_map
            
            # 3. Update Visualizer
            # Only update cloud if we have a depth map
            if 'current_depth' in locals():
                vis.update_cloud(frame, current_depth, mask=motion_mask)
                
            # 4. Update Camera View based on gestures
            if hand_state["is_tracking"]:
                vis.update_view(hand_state)

            # 5. Show Debug Frame
            cv2.imshow("Hand Tracking Debug", display_frame)
            
            # Check for close or reset
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key == ord('r'):
                vis.reset_view()
            elif key == ord('c'):
                vis.toggle_color()
            elif key == ord('b'):
                use_bg_sub = not use_bg_sub
                print(f"Motion Filter: {use_bg_sub}")
            elif key == ord('i'):
                vis.toggle_depth_inversion()
                
            if not vis.step():
                break

            frame_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        vis.close()

if __name__ == "__main__":
    main()
