from pypylon import pylon
import cv2
import os

# --- CONFIGURATION ---
CHECKERBOARD = (9, 7)   # (Columns, Rows) of INNER corners
SAVE_DIR = 'calib_images'
DISPLAY_SCALE = 0.3     # Change this to fit your screen
# ---------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# Connect to the Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print(f"Starting Highly Optimized Capture (Display scaled to {DISPLAY_SCALE*100}%)...")
print("Move the board around. Press 'C' to save when you see the colored lines.")
print("Press 'Q' to quit.")

count = 0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        
        # 1. THE ORIGINAL HIGH-RES IMAGE (We keep this safe to save later)
        img = image.GetArray() 
        
        # 2. RESIZE FIRST to save massive amounts of CPU processing
        h, w = img.shape[:2]
        new_w = int(w * DISPLAY_SCALE)
        new_h = int(h * DISPLAY_SCALE)
        display_img = cv2.resize(img, (new_w, new_h))
        
        # 3. Do all the heavy math on the SMALL image
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)
        
        if ret:
            # Draw on the small image
            cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners, ret)
            cv2.putText(display_img, "BOARD DETECTED - READY", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "Searching...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the lightweight video feed
        cv2.imshow('Smart Calibration Capture', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 4. Save the ORIGINAL 'img', NOT the resized one
        if key == ord('c') and ret:
            filename = os.path.join(SAVE_DIR, f"calib_{count:03d}.jpg")
            cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"Captured: {filename} (Saved at FULL resolution)")
            count += 1
        elif key == ord('c') and not ret:
            print("Cannot capture - board not detected clearly!")
            
        elif key == ord('q'):
            break

    grabResult.Release()

camera.StopGrabbing()
cv2.destroyAllWindows()