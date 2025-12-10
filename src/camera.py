import cv2
import threading
import time

class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.device_id}")
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Camera started on device {self.device_id}")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        print("Camera stopped")

if __name__ == "__main__":
    cam = Camera()
    cam.start()
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Test", frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
