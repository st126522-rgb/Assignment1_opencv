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

# Global state stored in a dict
state = {
    "keyboard_option": "",
    "contrast": 1.0,
    "brightness": 0,
    "kernel_size_gaussian": 5,
    "sigma": 0.5,
    "diameter_bilateral": 9,
    "sigmaColor": 75,
    "sigmaSpace": 75,
    "upper_canny": 100,
    "lower_canny": 200,
    "fps": 20,
    "panorama_frames": [],   # store frames for panorama
    "menu_active": False
}


def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def contrast_brightness(frame):
    c = state["contrast"]
    b = state["brightness"]
    adjusted_frame = np.clip(c * frame + b, 0, 255).astype(np.uint8)
    return adjusted_frame

def img_histogram(frame):
    chans = cv2.split(frame)
    colors = ("b", "g", "r")
    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.show()

def gaussian_filter(frame):
    return cv2.GaussianBlur(frame, (state["kernel_size_gaussian"], state["kernel_size_gaussian"]), state["sigma"])

def bilateral_filter(frame):
    return cv2.bilateralFilter(frame, state["diameter_bilateral"], state["sigmaColor"], state["sigmaSpace"])

def canny_edge(frame):
    return cv2.Canny(frame, state["upper_canny"], state["lower_canny"])

def hough_line_detection(frame):
    edge_detected = canny_edge(frame)
    lines = cv2.HoughLines(edge_detected, 1, np.pi / 180, 150)
    if lines is not None:
        for rho, theta in lines[:,0]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    return frame

def add_panorama_frame(frame):
    """Store frames for panorama stitching"""
    state["panorama_frames"].append(frame.copy())
    print(f"[INFO] Added frame {len(state['panorama_frames'])} for panorama.")

def panorama():
    """Simple panorama stitching using homography"""
    if len(state["panorama_frames"]) < 2:
        print("[WARN] Need at least 2 frames for panorama")
        return None

    # naive stitcher: just horizontally concat for now
    stitched = state["panorama_frames"][0]
    for i in range(1, len(state["panorama_frames"])):
        stitched = np.hstack((stitched, state["panorama_frames"][i]))
    return stitched

#function mapping 

filter_dict = {
    "g": gray_scale,
    "h": hough_line_detection,
    "b": gaussian_filter,
    "c": bilateral_filter,
    "e": canny_edge,
    "r": contrast_brightness,
    "s": hsv
}

#UI

def cam_options(frame):
    key = state["keyboard_option"]

    if key in filter_dict:
        return filter_dict[key](frame)

    elif key == "p":  # add panorama frame
        add_panorama_frame(frame)
        state["keyboard_option"] = ""  # reset to avoid continuous adding

    elif key == "o":  # show panorama
        pano = panorama()
        if pano is not None:
            cv2.imshow("Panorama", pano)
        state["keyboard_option"] = ""

    return frame



def run_cam():
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # key pressed
            state["keyboard_option"] = chr(key)

            if state["keyboard_option"] == "m":
                state["menu_active"] = not state["menu_active"]
                state["keyboard_option"] = ""

            if state["keyboard_option"] == "q":
                break

        # apply filter
        new_frame = cam_options(frame)

        # Overlay menu
        cv2.putText(new_frame, f"FPS: {state['fps']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if state["menu_active"]:
            cv2.putText(new_frame, "Keys: g=Gray | h=Hough | b=Gauss | c=Bilateral | e=Canny | r=Contrast | s=HSV",
                        (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(new_frame, "p=Add Panorama Frame | o=Show Panorama | q=Quit",
                        (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(new_frame, "Mennu[M] | Quit[Q]",
                        (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Camera", new_frame)

    cam.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    loaded_test_image=cv2.imread("test.jpg") #image for general testing
    road_img=cv2.imread("road.jpg") # image for hough line detection  
    # cv2.imshow("Test Image",loaded_test_image)
    # blurred=gaussian_filter(loaded_test_image,5,5)
    # test=hough_line_detection(road_img)
    # cv2.imshow("Tested Image",test)
    # cv2.waitKey(0)
    run_cam()
    # cv2.destroyAllWindows()