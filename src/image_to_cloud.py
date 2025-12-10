import cv2
import argparse
import open3d as o3d
import numpy as np
from depth import DepthEstimator
from visualizer import Visualizer
from camera import Camera
from hands import HandTracker

def main():
    parser = argparse.ArgumentParser(description="Static Image to Depth Cloud with Gesture Control")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--save", type=str, help="Optional path to save .ply file")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID for control")
    parser.add_argument("--fov_scale", type=float, default=0.6, help="Field of View scale (0.5=Wide/90deg, 1.0=Zoom/53deg). Default is 0.6")
    args = parser.parse_args()

    # 1. Load Image
    print(f"Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    # 2. Estimate Depth & Create Cloud
    print("Estimating Depth...")
    depth_est = DepthEstimator()
    depth_map = depth_est.predict(image)

    h, w = image.shape[:2]
    # Calculate intrinsics based on FOV scale
    # Smaller fx/fy = Wider FOV
    f = w * args.fov_scale
    metric_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h, fx=f, fy=f, cx=w/2, cy=h/2
    )
    pcd = Visualizer.create_pcd(image, depth_map, metric_intrinsic)

    if args.save:
        print(f"Saving to {args.save}...")
        o3d.io.write_point_cloud(args.save, pcd)

    # 3. Setup Interactive Control
    print("Starting Gesture Control (Press ESC to quit)...")
    cam = Camera(device_id=args.camera)
    cam.start()
    hands = HandTracker()
    vis = Visualizer(width=int(w * 1.5), height=int(h * 1.5))
    
    # Inject static cloud into visualizer
    vis.pcd = pcd
    vis.vis.add_geometry(pcd)
    vis.geom_added = True
    
    try:
        while True:
            # Control Loop
            frame = cam.get_frame()
            if frame is None:
                continue
                
            # Track Hands
            hand_state = hands.process(frame)
            
            # Update View
            if hand_state["is_tracking"]:
                vis.update_view(hand_state)
            
            # Step Visualizer
            if not vis.step():
                break
                
            # Show Camera Feed
            hands.draw_debug(frame, hand_state.get("landmarks"))
            cv2.imshow("Hand Control", frame)
            
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('r'):
                vis.reset_view()
            elif key == ord('c'):
                vis.toggle_color()
            elif key == ord('i'):
                vis.toggle_depth_inversion()
                vis.update_cloud(image, depth_map)
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        vis.close()

if __name__ == "__main__":
    main()
