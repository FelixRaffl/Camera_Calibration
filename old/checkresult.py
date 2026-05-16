from pypylon import pylon
import cv2
import numpy as np

# Load your new calibration data
with np.load('basler_calib_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

DISPLAY_SCALE = 0.4 # Will now comfortably fit your screen since it's only one image

# Connect to the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed

print("Live Undistortion Started!")
print("-> Press SPACEBAR to flip between RAW and CALIBRATED.")
print("-> Press 'Q' to quit.")

show_calibrated = True # Start by showing the good, flat image

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        
        # Determine which image to show based on the spacebar toggle
        if show_calibrated:
            # Apply the mathematical correction
            display_img = cv2.undistort(img, mtx, dist, None, mtx)
            label = "CALIBRATED / FLAT (Press SPACE to toggle)"
            color = (0, 255, 0) # Green text
        else:
            # Show the raw image
            display_img = img.copy()
            label = "RAW / DISTORTED (Press SPACE to toggle)"
            color = (0, 0, 255) # Red text
            
        # Resize to fit your screen
        h, w = display_img.shape[:2]
        new_w, new_h = int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)
        display_img_small = cv2.resize(display_img, (new_w, new_h))
        
        # Draw the label so you know what you are looking at
        cv2.putText(display_img_small, label, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Calibration Proof', display_img_small)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'Q' to quit
        if key == ord('q'):
            break
        # Press SPACEBAR to flip the image
        elif key == ord(' '): 
            show_calibrated = not show_calibrated

    grabResult.Release()

camera.StopGrabbing()
cv2.destroyAllWindows()