import cv2
import numpy as np
import glob
import os

def get_calibration_points(image_dir, image_format='jpg', grid_size=(9, 6), square_size=1.0):
    """
    Reads checkerboard images and extracts object points and image points.
    
    Args:
        image_dir (str): Directory containing the calibration images.
        image_format (str): Image extension (e.g., 'jpg', 'png').
        grid_size (tuple): Number of inner corners per a chessboard row and column.
        square_size (float): Real-world size of a chessboard square (e.g., in cm or mm).
        
    Returns:
        objpoints (list): 3D points in real world space.
        imgpoints (list): 2D points in image plane.
        image_shape (tuple): Width and height of the images.
    """
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    search_path = os.path.join(image_dir, f'*.{image_format}')
    images = glob.glob(search_path)

    if not images:
        raise ValueError(f"No images found in {image_dir} with format {image_format}")

    image_shape = None

    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_shape is None:
            image_shape = gray.shape[::-1] # (width, height)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)

    return objpoints, imgpoints, image_shape

def calibrate_camera(objpoints, imgpoints, image_shape):
    """
    Performs the camera calibration.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    return ret, mtx, dist, rvecs, tvecs

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """
    Calculates the mean reprojection error to assess calibration quality.
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        
    return mean_error / len(objpoints)

def undistort_image(image_path, mtx, dist):
    """
    Removes lens distortion from an image using the calibration matrices.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Refine the camera matrix to prevent black pixels around the edges
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image to the valid region of interest
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return img, dst