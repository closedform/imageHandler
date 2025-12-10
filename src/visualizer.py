import cv2
import open3d as o3d
import numpy as np
import threading

class Visualizer:
    def __init__(self, width=640, height=480):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Depth Hand Control", width=width, height=height)
        
        # Render Options for "More Dots" look
        opt = self.vis.get_render_option()
        opt.point_size = 3.0 # Finer particles
        opt.background_color = np.asarray([0.0, 0.0, 0.0]) # Pure Black background
        opt.show_coordinate_frame = True
        
        self.pcd = o3d.geometry.PointCloud()
        self.is_started = False
        
        # Camera intrinsics (approximate)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, 
            fx=width, fy=width, 
            cx=width/2, cy=height/2
        )
        
        self.view_control = None
        self.geom_added = False
        
        # Add a reference Grid
        self.grid = self.create_grid()
        self.vis.add_geometry(self.grid)
        
        # Color Toggle State
        self.use_color = True
        self.saved_colors = None
        
        # Depth Inversion State
        self.invert_depth = False
        self.current_zoom = 1.1  # Initial zoom (Backing off: higher = further)

    @staticmethod
    def create_grid(size=2000, step=200):
        # Create a simple grid on the X-Y plane at Z=2000 (Backdrop)
        lines = []
        points = []
        # Vertical lines
        for x in range(-size, size+1, step):
            points.append([x, -size, 1500])
            points.append([x, size, 1500])
            lines.append([len(points)-2, len(points)-1])
        
        # Horizontal lines
        for y in range(-size, size+1, step):
            points.append([-size, y, 1500])
            points.append([size, y, 1500])
            lines.append([len(points)-2, len(points)-1])
            
        colors = [[0.3, 0.3, 0.3] for _ in range(len(lines))]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    @staticmethod
    def create_pcd(rgb, depth, intrinsic, mask=None, invert=True):
        # Depth Anything V2 outputs relative "closeness" (high = close).
        # Open3D Expects Z-distance (high = far).
        
        # 1. Invert: (Max - val) makes close=low_val, far=high_val
        if invert:
            inv_depth = depth.max() - depth
        else:
            inv_depth = depth
        
        # 2. Normalize to 0-1 scale
        if inv_depth.max() > inv_depth.min():
            norm_depth = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min())
        else:
            norm_depth = inv_depth * 0 # Flat
            
        # 3. Map to World Z units (e.g. 0 to 1000)
        # We add a bias (e.g. 50) so the closest point isn't inside the camera
        depth_scale = 1000.0 
        z_depth = (norm_depth * depth_scale) + 50.0
        
        # 4. Apply Mask if provided (remove static/masked pixels)
        if mask is not None:
             # resizing mask to depth size if needed (though we assume same size)
             if mask.shape[:2] != z_depth.shape[:2]:
                 mask = cv2.resize(mask, (z_depth.shape[1], z_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
             
             # Set invalid depth to 0 (ignored by Open3D)
             z_depth[mask == 0] = 0.0

        # Efficiently create RGBD image
        color_raw = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_raw = o3d.geometry.Image(z_depth.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, 
            convert_rgb_to_intensity=False,
            depth_scale=1.0, 
            depth_trunc=20000.0 # No truncation!
        )
        
        pcd_new = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic
        )
        # Flip to look right
        pcd_new.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return pcd_new

    def toggle_color(self):
        self.use_color = not self.use_color
        
        if self.pcd is None or not self.geom_added:
            return

        if not self.use_color:
            # Switch to Hologram Mode (Bright particles)
            # Save existing colors first
            self.saved_colors = np.asarray(self.pcd.colors).copy()
            self.pcd.paint_uniform_color([0.85, 0.85, 0.85])  # Bright white-ish particles
        else:
            # Restore Color
            if self.saved_colors is not None and len(self.saved_colors) == len(self.pcd.points):
                self.pcd.colors = o3d.utility.Vector3dVector(self.saved_colors)
                # self.saved_colors = None # Support multiple toggles
                
        self.vis.update_geometry(self.pcd)

    def toggle_depth_inversion(self):
        self.invert_depth = not self.invert_depth
        print(f"Depth Inversion: {self.invert_depth}")

    def update_cloud(self, rgb, depth, mask=None):
        pcd_new = self.create_pcd(rgb, depth, self.intrinsic, mask=mask, invert=self.invert_depth)
        
        # Apply color preference
        if not self.use_color:
            pcd_new.paint_uniform_color([0.85, 0.85, 0.85])

        if not self.geom_added:
            self.pcd = pcd_new
            self.vis.add_geometry(self.pcd)
            self.view_control = self.vis.get_view_control()
            self.reset_view()
            self.geom_added = True
        else:
            self.pcd.points = pcd_new.points
            # If we are in "Color Mode", we use the new frame's colors.
            # If we are in "White Mode", we use white (already painted above).
            self.pcd.colors = pcd_new.colors
            self.vis.update_geometry(self.pcd)

    def reset_view(self):
        if not self.view_control:
            return

        targets = []
        if self.pcd is not None and len(self.pcd.points) > 0:
            targets.append(np.asarray(self.pcd.points))
        if self.grid is not None:
            targets.append(np.asarray(self.grid.points))

        if targets:
            all_pts = np.vstack(targets)
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(all_pts)
            )
            center = bbox.get_center()
        else:
            center = np.array([0.0, 0.0, 0.0])

        self.view_control.set_lookat(center)
        self.view_control.set_front([0, 0, -1])
        self.view_control.set_up([0, 1, 0])
        self.current_zoom = 1.1
        self.view_control.set_zoom(self.current_zoom)

        # Ensure both grid and cloud stay registered after reset
        if self.grid is not None:
            self.vis.update_geometry(self.grid)
        if self.pcd is not None and self.geom_added:
            self.vis.update_geometry(self.pcd)

    def update_view(self, gesture_state):
        if not self.view_control:
            return
            
        gesture = gesture_state.get("gesture")
        if gesture == "ROTATE":
            # Rotation from hand movement (x, y)
            dx, dy = gesture_state.get("delta", (0, 0))
            scale = 800.0 
            self.view_control.rotate(dx * scale, dy * scale)
            
            # Legacy: specific zoom based on hand depth is removed.
            # Zoom from hand depth (forward/backward movement)
            # zoom_delta = gesture_state.get("zoom_delta", 0.0)
            # if zoom_delta != 0.0:
            #     # ... logic removed ...
            #     pass
            
        elif gesture == "PAN":
            dx, dy = gesture_state.get("delta", (0, 0))
            scale = 800.0 
            self.view_control.translate(dx * scale, dy * scale)
            
        elif gesture == "ZOOM":
            scale_factor = gesture_state.get("scale_factor", 1.0)
            if scale_factor != 1.0:
                 # Sensitivity Factor
                 sensitivity = 0.5 # For direct value subtraction, this is actually quite strong if per-frame
                 
                 # scale_factor > 1 (Spreading) -> Want Zoom IN -> Decrease current_zoom
                 # scale_factor < 1 (Pinching) -> Want Zoom OUT -> Increase current_zoom
                 
                 # Calculate deviation
                 diff = scale_factor - 1.0
                 
                 # Apply updates
                 # If spreading (diff > 0), we want to subtract from zoom
                 # If pinching (diff < 0), we want to add to zoom
                 self.current_zoom -= (diff * sensitivity)
                 
                 # Clamp values
                 self.current_zoom = max(0.2, min(3.0, self.current_zoom))
                 
                 print(f"Zoom Level: {self.current_zoom:.3f} (Factor {scale_factor:.3f})")
                 self.view_control.set_zoom(self.current_zoom)

    def step(self):
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def close(self):
        self.vis.destroy_window()
