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

    def clear_controls(self):
        try:
            cv2.destroyWindow("Controls")
        except cv2.error:
            pass  # Ignore if Controls is not open

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
        else:
            return frame

    # ---------------- Menu Display ----------------
    def draw_menu(self, frame):
        h, w = frame.shape[:2]
        menu_text = "g: Gaussian | b: Bilateral | c: Canny | y: Gray | h: HSV | m: Remove | q: Quit"
        cv2.putText(frame, menu_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

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