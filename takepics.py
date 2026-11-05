"""Live capture of checkerboard images for the camera calibration.

Press C to save a full-resolution image while the board is detected, Q to quit.
"""
import glob
import os

import cv2
from pypylon import pylon

CHECKERBOARD = (9, 7)  # (columns, rows) of INNER corners
SAVE_DIR = 'calib_images'
DISPLAY_SCALE = 0.3  # Change this to fit your screen
FIND_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
              + cv2.CALIB_CB_FAST_CHECK)


def next_image_number():
    """Return the first number above all saved images, so nothing is overwritten."""
    numbers = [-1]
    for path in glob.glob(os.path.join(SAVE_DIR, 'calib_*.jpg')):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.removeprefix('calib_')
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers) + 1


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    count = next_image_number()

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    print("Move the board around. Press 'C' to save when you see the colored lines.")
    print("Press 'Q' to quit.")

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if not grab_result.GrabSucceeded():
                    continue
                img = converter.Convert(grab_result).GetArray()

                # Detect on the small preview, it is much faster than the full image
                h, w = img.shape[:2]
                display_img = cv2.resize(img, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
                gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=FIND_FLAGS)

                if found:
                    cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners, found)
                    cv2.putText(display_img, "BOARD DETECTED - READY", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "Searching...", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Smart Calibration Capture', display_img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c') and found:
                    filename = os.path.join(SAVE_DIR, f"calib_{count:03d}.jpg")
                    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f"Captured: {filename} (Saved at FULL resolution)")
                    count += 1
                elif key == ord('c'):
                    print("Cannot capture - board not detected clearly!")
                elif key == ord('q'):
                    break
            finally:
                grab_result.Release()
    finally:
        camera.StopGrabbing()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
