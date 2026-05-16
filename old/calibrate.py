import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
CHECKERBOARD = (9, 7)  # Make sure this matches your capture script
SQUARE_SIZE = 0.01     # <--- UPDATE THIS: Size of one square in METERS (e.g., 0.025)
IMAGE_DIR = 'calib_images/*.jpg'
# ---------------------

if SQUARE_SIZE == 0.000:
    print("ERROR: You forgot to update the SQUARE_SIZE variable at the top!")
    exit()

# Termination criteria for sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points based on the real-world dimensions
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob(IMAGE_DIR)
img_shape = None

print(f"Found {len(images)} images. Starting heavy processing...\n")

successful_images = 0

for i, fname in enumerate(images):
    filename = os.path.basename(fname)
    print(f"[{i+1}/{len(images)}] Scanning {filename}...", end=" ", flush=True)
    
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = gray.shape[::-1]
    
    # --- THE HIGH-RESOLUTION FIX ---
    # These flags clean up lighting and noise before looking for the corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    
    if ret:
        objpoints.append(objp)
        # Refine corner locations to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_refined)
        successful_images += 1
        print("Corners found!")
    else:
        print("FAILED to find all corners.")

print(f"\nSuccessfully extracted corners from {successful_images} out of {len(images)} images.")

if successful_images == 0:
    print("\nCRITICAL ERROR: Could not find the board in ANY image.")
    print("Fix 1: Try changing CHECKERBOARD to (10, 8) instead of (8, 10).")
    print("Fix 2: Ensure your images aren't too blurry or poorly lit.")
    exit()

print("Calculating final camera matrix and distortion... (Almost done!)")

# Perform the actual calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

print("\n" + "="*30)
print("--- CALIBRATION RESULTS ---")
print("="*30)
print(f"Reprojection Error: {ret:.4f} pixels")
print("\nCamera Matrix (Intrinsics):\n", mtx)
print("\nDistortion Coefficients:\n", dist.ravel())

# Save the data for your live application
np.savez('basler_calib_data.npz', mtx=mtx, dist=dist)
print("\nSuccess! Data saved to 'basler_calib_data.npz'")