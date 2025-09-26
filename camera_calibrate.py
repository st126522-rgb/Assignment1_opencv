# calibrate_camera.py
#
# Description:
# This script calibrates the user's webcam using a chessboard pattern.
#
# Instructions:
# 1. Print a 9x6 chessboard pattern and mount it on a flat surface.
# 2. Run this script.
# 3. Show the chessboard to the camera from various angles and distances.
# 4. The script will automatically collect ~20 good views and then compute the camera matrix
#    and distortion coefficients.
# 5. The results will be saved to "calibration.npz".
#
# This script directly relates to Week 5: Camera Calibration.

import numpy as np
import cv2
import time

# --- Configuration ---
# You can change these values to match your chessboard
CHESSBOARD_SIZE = (9, 6)  # Number of internal corners (width, height)
SQUARE_SIZE_MM = 25       # The real-world size of a square on your chessboard in mm

# --- Calibration Setup ---
# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)               # Create a grid of points in 3D space 9*6 = 54 points
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) 

# np.mgrid creates a mesh grid:
# [(0,0), (1,0), (2,0), … (8,0),
#  (0,1), (1,1), …        (8,1),
#  ...
#  ...                    (8,5)]

# Reshape and assign to objp's first two columns (x, y coordinates)
# The z coordinate remains 0 since the chessboard is flat

# objp now is:
# [[0. 0. 0.]
#  [1. 0. 0.]
#  [2. 0. 0.]
#  ...
#  [8. 5. 0.]]

# Scale object points to actual square size in mm
objp = objp * SQUARE_SIZE_MM

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images_captured = 0
TARGET_IMAGES = 20 # Number of images to capture for calibration

# --- Main Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Starting camera calibration...")
print(f"Show the {CHESSBOARD_SIZE} chessboard to the camera.")
print(f"Need to capture {TARGET_IMAGES} good views.")

last_capture_time = time.time()

while images_captured < TARGET_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)    
    # True if found, corners are the pixel coordinates of the corners Nx1x2

    display_frame = frame.copy()
    
    # If found, add object points, image points (after refining them)
    if ret_corners:
        # Draw the corners to give visual feedback
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners, ret_corners)

        # Capture an image every 2 seconds to allow for repositioning
        if time.time() - last_capture_time > 2:
            print(f"Found corners! Capturing image {images_captured + 1}/{TARGET_IMAGES}...")
            
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Refine the corner locations to sub-pixel accuracy  -- lower projection error
            # shape of corners2 is Nx1x2

            objpoints.append(objp)
            imgpoints.append(corners2)
            
            images_captured += 1
            last_capture_time = time.time()
    
    # Display status on the frame
    text = f"Images captured: {images_captured}/{TARGET_IMAGES}"
    cv2.putText(display_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Calibration', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nAll images captured. Performing calibration...")
cap.release()
cv2.destroyAllWindows()

# --- Perform Calibration ---
try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\nCalibration successful!")
        print("Camera Matrix (mtx):")
        print(mtx)
        print("\nDistortion Coefficients (dist):")
        print(dist)

        # Save the calibration result
        np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print('\nCalibration data saved to "calibration.npz"')

        # Calculate and display re-projection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print(f"Total re-projection error: {mean_error/len(objpoints):.4f}")

    else:
        print("Calibration failed. Please try again.")

except Exception as e:
    print(f"An error occurred during calibration: {e}")
    print("Not enough valid images. Please restart the script and try again.")