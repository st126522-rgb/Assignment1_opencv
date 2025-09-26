# Convert image color between RGB ↔ grey ↔ HSV
# Contrast and brightness
# Show image histogram
# Gaussian filter with changable parameter
# Bilerater filter with changeable parameters
# Canny edge detection
# Line detection using Hough Transform
# Create a panorama
# Do not use opencv function, write your own function to perform the panorama
# Image translation, rotation, and scale
# Calibrate the camera
# Augmented Reality
# Instead of projecting a simple cube, use the provided .OBJ file to project the TREX model in AR mode.
# The initial size of model will be small, so increase the size of model as well.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os

class CameraApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.active_filter = None

        # Gaussian params
        self.gaussian_kernel = 5
        self.gaussian_sigma = 1

        # Bilateral params
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75

        # Canny params
        self.canny_th1 = 100
        self.canny_th2 = 200

        self.panorama_frames = []  # store frames for panorama
        
        #Transform params
        self.trans_x = 0
        self.trans_y = 0
        self.angle = 0
        self.scale = 1.0

        # Load T-Rex model for AR
        self.trex_model = self._load_obj('trex_model.obj')

        # --- ARUCO setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size=1
        
        # Camera calibration load
        self.mtx = None
        self.dist = None
        self._load_calibration()
        
       
            
    # ---------------- Trackbar Callbacks ----------------
    def update_gaussian_kernel(self, val):
        if val % 2 == 0:
            val += 1
        self.gaussian_kernel = max(1, val)

    def update_gaussian_sigma(self, val):
        self.gaussian_sigma = max(1, val)

    def update_bilateral_d(self, val):
        self.bilateral_d = max(1, val)

    def update_bilateral_sigma_color(self, val):
        self.bilateral_sigma_color = max(1, val)

    def update_bilateral_sigma_space(self, val):
        self.bilateral_sigma_space = max(1, val)

    def update_canny_th1(self, val):
        self.canny_th1 = val

    def update_canny_th2(self, val):
        self.canny_th2 = val
        
    def update_trans_x(self, val):
        self.trans_x = val - 100  # range [-100, 100]

    def update_trans_y(self, val):
        self.trans_y = val - 100  # range [-100, 100]

    def update_angle(self, val):
        self.angle = val - 180  # range [-180, 180]

    def update_scale(self, val):
        self.scale = max(val / 100.0, 0.01)  # range [0.01, 3.0]


    # ---------------- Trackbar Setup ----------------
    def setup_gaussian_controls(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Kernel", "Controls", self.gaussian_kernel, 31, self.update_gaussian_kernel)
        cv2.createTrackbar("Sigma", "Controls", self.gaussian_sigma, 20, self.update_gaussian_sigma)

    def setup_bilateral_controls(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Diameter", "Controls", self.bilateral_d, 20, self.update_bilateral_d)
        cv2.createTrackbar("Sigma Color", "Controls", self.bilateral_sigma_color, 150, self.update_bilateral_sigma_color)
        cv2.createTrackbar("Sigma Space", "Controls", self.bilateral_sigma_space, 150, self.update_bilateral_sigma_space)

    def setup_canny_controls(self):
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Threshold1", "Controls", self.canny_th1, 500, self.update_canny_th1)
        cv2.createTrackbar("Threshold2", "Controls", self.canny_th2, 500, self.update_canny_th2)
        
    def setup_transform_controls(self):
        cv2.namedWindow("Controls")
        cv2.createTrackbar("Trans X", "Controls", 100, 200, self.update_trans_x)
        cv2.createTrackbar("Trans Y", "Controls", 100, 200, self.update_trans_y)
        cv2.createTrackbar("Rotation", "Controls", 180, 360, self.update_angle)
        cv2.createTrackbar("Scale", "Controls", 100, 300, self.update_scale)

        
      # ---------- PANORAMA STITCHING ----------
    def stitch_frames(self):
        if len(self.panorama_frames) < 2:
            print("Need at least 2 frames for panorama")
            return None

        # start with the first frame
        stitched = self.panorama_frames[0]
        for i in range(1, len(self.panorama_frames)):
            stitched = self.stitch_pair(stitched, self.panorama_frames[i])
            if stitched is None:
                print("Stitching failed at frame", i)
                break
        return stitched

    def stitch_pair(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        w = img1.shape[1] + img2.shape[1]
        h = max(img1.shape[0], img2.shape[0])

        result = cv2.warpPerspective(img2, H, (w, h))
        result[0:img1.shape[0], 0:img1.shape[1]] = img1
        return result

    def clear_controls(self):
        try:
            cv2.destroyWindow("Controls")
        except cv2.error:
            pass  # Ignore if Controls is not open
        
    def show_histogram(self, frame):
        # If frame is color, split channels
        if len(frame.shape) == 3:
            chans = cv2.split(frame)
            colors = ("b", "g", "r")
            plt.figure(figsize=(8,4))
            for chan, color in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [256], [0,256])
                plt.plot(hist, color=color)
            plt.title("Color Histogram")
            plt.xlabel("Pixel value")
            plt.ylabel("Frequency")
            plt.show()
        else:
            # Grayscale
            hist = cv2.calcHist([frame], [0], None, [256], [0,256])
            plt.figure(figsize=(8,4))
            plt.plot(hist, color="k")
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel value")
            plt.ylabel("Frequency")
            plt.show()
            
    def _load_calibration(self):
        if os.path.exists("calibration.npz"):
            with np.load("calibration.npz") as X:
                self.mtx = X["mtx"]
                self.dist = X["dist"]
            print("Calibration data loaded.")
        else:
            self.mtx = np.eye(3)
            self.dist = np.zeros((1,5))
            print("Calibration data missing, using default values.")
            
                
    def _load_obj(self, filename):        
    
        verts = []
        faces = []

        if not os.path.exists(filename):
            print(f"[OBJ] file not found: {filename}")
            return None, None

        with open(filename, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, z = map(float, parts[1:4])
                        verts.append([x, y, z])
                    except:
                        continue
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    face = []
                    for p in parts:
                        # handle formats like "f v", "f v/vt", "f v/vt/vn"
                        idx = p.split("/")[0]
                        try:
                            face.append(int(idx) - 1)
                        except:
                            pass
                    if len(face) >= 3:
                        # optionally triangulate polygons (fan triangulation)
                        if len(face) > 3:
                            for i in range(1, len(face)-1):
                                faces.append([face[0], face[i], face[i+1]])
                        else:
                            faces.append(face)

        if len(verts) == 0 or len(faces) == 0:
            print(f"[OBJ] Warning: loaded but empty verts/faces from {filename}")
            return None, None

        verts = np.array(verts, dtype=np.float32)
        print(f"[OBJ] Loaded {filename}: vertices={len(verts)}, faces={len(faces)}")
        return verts, faces


    
    def calibrate_camera(self):
        CHESSBOARD_SIZE = (9,6)
        SQUARE_SIZE_MM = 25
        objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)
        objp *= SQUARE_SIZE_MM
        objpoints, imgpoints = [], []
        images_captured = 0
        TARGET_IMAGES = 20
        last_capture_time = time.time()

        print("Starting camera calibration...")
        while images_captured < TARGET_IMAGES:
            ret, frame = self.cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            display_frame = frame.copy()
            if ret_corners and (time.time() - last_capture_time > 2):
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)
                images_captured += 1
                last_capture_time = time.time()
            text = f"Images captured: {images_captured}/{TARGET_IMAGES}"
            cv2.putText(display_frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            cv2.imshow("Calibration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('m'): break

        cv2.destroyWindow("Calibration")
        # Compute calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print("Calibration saved to 'calibration.npz'")


        
    def _draw_ar_model(self, frame):
        cv2.putText(frame, "Mode: AUGMENTED REALITY", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        verts, faces = self.trex_model if self.trex_model else (None, None)
        if verts is None or faces is None:
            cv2.putText(frame, "T-Rex model missing", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            cv2.putText(frame, "No ArUco marker detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return frame

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Pose estimation
        marker_size_m = 0.15
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_m, self.mtx, self.dist)

        # Draw axis for first marker
        cv2.drawFrameAxes(frame, self.mtx, self.dist, rvecs[0], tvecs[0], marker_size_m * 0.5)

        # Project T-Rex on first marker
        frame = self._project_obj(frame, verts, faces, rvecs[0], tvecs[0])
        return frame


    def _project_obj(self, frame, vertices, faces, rvec, tvec):
        if vertices is None or faces is None:
            return frame

        verts = vertices.copy()

        # --- Normalize and scale ---
        verts -= np.mean(verts, axis=0)  # center at origin
        scale = np.max(np.linalg.norm(verts, axis=1))
        verts /= scale
        verts *= 0.2  # adjust size for visibility

        # --- Rotate model: X-axis -90° ---
        Rx = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]], dtype=float)
        verts = verts @ Rx.T

        # --- Optional: lift slightly above origin ---
        # verts[:, 1] += 0.0  # keep at exact origin

        # --- Project to 2D ---
        imgpts, _ = cv2.projectPoints(verts, rvec, tvec, self.mtx, self.dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw faces
        for face in faces:
            pts = imgpts[face]
            cv2.fillConvexPoly(frame, pts, (0, 255, 0))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return frame



   
    # ---------------- Filter Application ----------------
    def apply_filter(self, frame):
        if self.active_filter == "g":  # Gaussian
            return cv2.GaussianBlur(frame, (self.gaussian_kernel, self.gaussian_kernel), self.gaussian_sigma)
        elif self.active_filter == "b":  # Bilateral
            return cv2.bilateralFilter(frame, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
        elif self.active_filter == "c":  # Canny
            return cv2.Canny(frame, self.canny_th1, self.canny_th2)
        elif self.active_filter == "gray":  # Grayscale
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.active_filter == "hsv":  # HSV
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.active_filter == "t":
            return self.apply_transform(frame)
        elif self.active_filter == "ar":
            return self._draw_ar_model(frame)
        else:
            return frame

    def apply_transform(self, frame):
        h, w = frame.shape[:2]
        # Translation matrix
        M_trans = np.float32([[1, 0, self.trans_x], [0, 1, self.trans_y]])
        frame = cv2.warpAffine(frame, M_trans, (w, h))
        
        # Rotation + Scale
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), self.angle, self.scale)
        frame = cv2.warpAffine(frame, M_rot, (w, h))
        
        return frame

        
    
    # ---------------- Menu Display ----------------
    def draw_menu(self, frame):
        h, w = frame.shape[:2]
        lines = [
            "Filters: [G] Gaussian | [B] Bilateral | [C] Canny | [Y] Gray | [H] HSV | [M] Remove Filter/ None",
            "[I] Histogram | [T] Transform | [K] Calibrate Camera | [A] AR Mode",
            "[P] Add Frame | [O] Stitch | [X] Clear Frames | [Q] Quit"
        ]
        y0 = h - 60
        for i, text in enumerate(lines):
            y = y0 + i * 20
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # ---------------- Main Loop ----------------
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            output = self.apply_filter(frame)
            self.draw_menu(output)
            cv2.imshow("Camera", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Gaussian
            elif key == ord("g"):
                if self.active_filter == "g":
                    self.active_filter = None
                    self.clear_controls()
                else:
                    self.active_filter = "g"
                    self.clear_controls()
                    self.setup_gaussian_controls()

            #histogram
            elif key == ord("i"):
                self.show_histogram(frame)
                
            # Bilateral
            elif key == ord("b"):
                if self.active_filter == "b":
                    self.active_filter = None
                    self.clear_controls()
                else:
                    self.active_filter = "b"
                    self.clear_controls()
                    self.setup_bilateral_controls()

            # Canny
            elif key == ord("c"):
                if self.active_filter == "c":
                    self.active_filter = None
                    self.clear_controls()
                else:
                    self.active_filter = "c"
                    self.clear_controls()
                    self.setup_canny_controls()

            # Gray
            elif key == ord("y"):
                if self.active_filter == "gray":
                    self.active_filter = None
                else:
                    self.active_filter = "gray"
                self.clear_controls()  # no sliders

            # HSV
            elif key == ord("h"):
                if self.active_filter == "hsv":
                    self.active_filter = None
                else:
                    self.active_filter = "hsv"
                self.clear_controls()  # no sliders
                
            # Panorama options
            elif key == ord("p"):  # add frame
                self.panorama_frames.append(frame.copy())
                print(f"Added frame {len(self.panorama_frames)} for panorama")
            elif key == ord("o"):  # stitch
                stitched = self.stitch_frames()
                if stitched is not None:
                    cv2.imshow("Panorama", stitched)
            elif key == ord("x"):  # clear frames
                self.panorama_frames = []
                print("Panorama list cleared")
                
            #Image Transformations
            elif key == ord("t"):
                self.active_filter = "t"
                self.clear_controls()
                self.setup_transform_controls()      
                
            #Calibrate Cam
            elif key == ord("k"):
                self.calibrate_camera()
            
            #AR mode  
            elif key == ord("a"):
                 self.active_filter = "ar"

            # Remove filter
            elif key == ord("m"):
                self.active_filter = None
                self.clear_controls()

        self.cap.release()
        cv2.destroyAllWindows()
    
if __name__=="__main__":
    loaded_test_image=cv2.imread("test.jpg") #image for general testing
    road_img=cv2.imread("road.jpg") # image for hough line detection  
    # cv2.imshow("Test Image",loaded_test_image)
    # blurred=gaussian_filter(loaded_test_image,5,5)
    # test=hough_line_detection(road_img)
    # cv2.imshow("Tested Image",test)
    # cv2.waitKey(0)
    app = CameraApp()
    app.run()
    # cv2.destroyAllWindows()