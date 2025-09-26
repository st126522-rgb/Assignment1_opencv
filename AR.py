
import cv2
import numpy as np
import os

def draw_text(frame, text, pos=(20, 50), scale=1.0, color=(255, 255, 255)):
    """Draws white text with a black outline for better visibility."""
    # The thickness is scaled with the font size for a consistent look
    thickness = int(scale * 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

class ComputerVisionDemo:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # --- General State ---
        self.mode = 'NORMAL'
        self.debug_mode = False
        self.window_name = 'CV Course Demo'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # --- AR & Pinhole Explorer State ---
        self.mtx = np.eye(3)
        self.dist = np.zeros((1, 5))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params) 
        self._load_calibration()
        self.pinhole_params = {'fx': self.mtx[0, 0], 'fy': self.mtx[1, 1], 'cx': self.mtx[0, 2], 'cy': self.mtx[1, 2]}

    def _load_calibration(self):
        if os.path.exists('calibration.npz'):
            with np.load('calibration.npz') as X:
                self.mtx, self.dist = [X[i] for i in ('mtx', 'dist')]
            print("Calibration data loaded.")
        else:
            print("WARNING: 'calibration.npz' not found. AR and Pinhole modes will not be accurate.")

    def _cleanup_ui(self):
        cv2.destroyWindow(self.window_name)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def _set_mode(self, new_mode):
        if self.mode != new_mode:
            self.mode = new_mode
            print(f"Switched to {self.mode} mode.")
            self._cleanup_ui()


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            display_frame = self._process_frame(frame.copy())

            help_text = "[A]R [N]ormal [Q]uit"
            draw_text(display_frame, help_text, pos=(10, display_frame.shape[0] - 20), scale=0.6)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if self._handle_key_press(key):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        mode_map = {
            'NORMAL': self._draw_normal_mode,
            'AR': self._draw_ar_mode,
        }
        handler = mode_map.get(self.mode, self._draw_normal_mode)
        return handler(frame)
    
    def _draw_normal_mode(self, frame):
        draw_text(frame, "Mode: NORMAL")
        return frame

    def _handle_key_press(self, key):
        if key == ord('q'): return True
        
        mode_keys = {
            ord('n'): 'NORMAL', ord('a'): 'AR'
        }
        if key in mode_keys:
            self._set_mode(mode_keys[key])
            return False

    # --- AR & Pinhole Implementations---
    def _draw_ar_cube(self, frame, camera_matrix):
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is not None:
            print(f"Detected ArUco markers: {ids.flatten()}")
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, self.dist)
            axis_len = 0.05
            obj_pts = np.float32([
                [0,0,0], [axis_len,0,0], [axis_len,axis_len,0], [0,axis_len,0],
                [0,0,-axis_len], [axis_len,0,-axis_len], [axis_len,axis_len,-axis_len], [0,axis_len,-axis_len]
            ])
            img_pts, _ = cv2.projectPoints(obj_pts, rvecs[0], tvecs[0], camera_matrix, self.dist)
            img_pts = np.int32(img_pts).reshape(-1, 2)
            # Define cube faces by indices
            faces = [
                [0, 1, 2, 3],  # bottom
                [4, 5, 6, 7],  # top
                [0, 1, 5, 4],  # side 1
                [1, 2, 6, 5],  # side 2
                [2, 3, 7, 6],  # side 3
                [3, 0, 4, 7],  # side 4
            ]
            face_colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]
            # Draw filled faces
            for idx, face in enumerate(faces):
                cv2.fillConvexPoly(frame, img_pts[face], face_colors[idx], lineType=cv2.LINE_AA)
            # Draw black wireframe
            for face in faces:
                cv2.polylines(frame, [img_pts[face]], isClosed=True, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA)
        else:
            print("No ArUco markers detected.")
            draw_text(frame, "No ArUco markers detected", pos=(20, 50), color=(0, 0, 255))
        frame = cv2.flip(frame, 1)  # Flip back to original orientation
        return frame

    def _draw_ar_mode(self, frame):
        draw_text(frame, "Mode: AUGMENTED REALITY")
        return self._draw_ar_cube(frame, self.mtx)


if __name__ == '__main__':
    app = ComputerVisionDemo()
    app.run()