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

### Capture video from camera and save it to a mp4 file.

def gray_scale(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

def hsv(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

def contrast_brightness(frame,contrast=1.0,brightness=0):
    adjusted_frame=np.clip(contrast*frame+brightness,0,255).astype(np.uint8)
    return adjusted_frame



# Open the default camera
def run_cam():
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
    loaded_test_image=cv2.imread(r"C:\Users\gaurav\OneDrive\Desktop\xtra\217398584_4157042961046193_1827823406268772815_n.jpg")
    cv2.imshow("Test Image",loaded_test_image)
    cv2.imshow("Tested Image",contrast_brightness(loaded_test_image,1.5,0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()