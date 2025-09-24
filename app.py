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
### Capture video from camera and save it to a mp4 file.
panaroma_frames=[]
contrast=1.0,brightness=0
kernel_size_gaussian=5, sigma=0.5
diamter_bilateral=9,sigmaColor=75,sigmaSpace=75
upper_canny=100,lower_canny=200
selected_option=""

def gray_scale(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

def hsv(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

def contrast_brightness(frame,contrast=contrast,brightness=brightness):
    adjusted_frame=np.clip(contrast*frame+brightness,0,255).astype(np.uint8)
    return adjusted_frame

def img_histogram(frame):
    B_histo = cv2.calcHist([frame],[0], None, [256], [0,256])
    G_histo = cv2.calcHist([frame],[1], None, [256], [0,256])
    R_histo = cv2.calcHist([frame],[2], None, [256], [0,256])
    return B_histo, G_histo, R_histo

def gaussian_filter(frame, kernel_size=kernel_size_gaussian, sigma=sigma):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)

def bilerater_filter(frame,diamter=diamter_bilateral,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace):
    return cv2.bilateralFilter(frame, diamter, sigmaColor, sigmaSpace)


def canny_edge(frame,upper=100,lower=200):
    return cv2.Canny(frame,upper,lower)

def hough_line_detection(frame):
    edge_detected_frame=canny_edge(frame)
    lines=cv2.HoughLines(edge_detected_frame,1,np.pi/180,150)
    if lines is not None:
        for i in range(0,len(lines)):
            rho=lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    return frame
        

def panaroma(frames):
    pass

def image_transform(frame,translation,rotation,scale):
    pass

def camera_calibration(frames):
    pass

def AR(frame):
    pass


def cam_options(selected):
    pass


def run_cam():
# Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        new_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        canny_frame=cv2.Canny(new_frame,100,200)
        cv2.imshow('Camera', canny_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    loaded_test_image=cv2.imread("test.jpg") #image for general testing
    road_img=cv2.imread("road.jpg") # image for hough line detection  
    # cv2.imshow("Test Image",loaded_test_image)
    # blurred=gaussian_filter(loaded_test_image,5,5)
    test=hough_line_detection(road_img)
    cv2.imshow("Tested Image",test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()