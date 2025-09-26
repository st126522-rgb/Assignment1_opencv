# CameraApp

**Description:**  
Real-Time Image Processing and AR application using Python and OpenCV. Allows filters, image transformations, panorama creation, camera calibration, and projection of a 3D T-Rex model in augmented reality.

---

## Installation

- **Python Version:** 3.8+
- **Required Packages:**
  - `opencv-python`
  - `opencv-contrib-python`
  - `numpy`
  - `matplotlib`
- **Required Files:**
  - `app.py`
  - `trex_model.obj`
  - Test images (optional)

---

## Running the App

- **Command:**  
  ```bash
  python app.py

## Keyboard Shortcuts / Controls

**Filters & Image Processing:**

- **[G]** : Gaussian Blur (adjustable via trackbars)  
- **[B]** : Bilateral Filter (adjustable via trackbars)  
- **[C]** : Canny Edge Detection (adjustable via trackbars)  
- **[Y]** : Grayscale Conversion  
- **[H]** : HSV Conversion  
- **[M]** : Remove Active Filter  

**Other Image Operations:**

- **[I]** : Show Histogram of current frame  
- **[T]** : Image Transformations (translation, rotation, scale via trackbars)  
- **[K]** : Calibrate Camera (opens calibration window)  
- **[A]** : AR Mode (projects T-Rex model on ArUco marker)  

**Panorama Controls:**

- **[P]** : Add current frame to panorama list  
- **[O]** : Stitch panorama frames and show result  
- **[X]** : Clear panorama frames  

**Exit / Quit:**

- **[Q]** : Quit application


## Camera Calibration

When you press **[K]**, a new window opens to capture chessboard images for calibration.

**How it works:**

- A chessboard of size 9x6 is used. Each square is assumed to be 25mm.  
- The application automatically detects corners in the frames.  
- It waits **2 seconds** between captures to avoid duplicates.  
- **Number of images required:** 20 (default).  
- During calibration, the number of captured images is displayed on the window.  

**Controls in Calibration Window:**

- **[M]** : Capture an image when chessboard corners are detected.  
- **[Q]** : Quit calibration window without completing.  

After capturing all required images, calibration parameters (camera matrix and distortion coefficients) are computed and saved in `calibration.npz`.  
If the file already exists, it will be loaded automatically when starting the app.


## Augmented Reality (AR) Mode

Press **[A]** to enter AR mode. The application projects a 3D T-Rex model onto an ArUco marker.

**Requirements:**

- Camera calibration must be done or `calibration.npz` must exist.
- The T-Rex model is loaded from `trex_model.obj`.
- An ArUco marker must be visible to the camera.

**How it works:**

- The AR mode detects ArUco markers in the frame.  
- If a marker is detected, it estimates its pose using the camera calibration data.  
- The T-Rex model is projected onto the marker based on the pose.  
- If no marker is detected, a warning message appears on the frame.

**Notes:**

- The initial size of the model is small. Scaling can be adjusted in transform controls if required.  
- The AR projection will fail if calibration is missing or marker is not visible.  
