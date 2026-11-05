"""Live view with focus peaking for aligning and focusing the Basler camera."""
import datetime

import cv2
from basler_easy_connect import EasyBasler


def resize_frame_keep_ratio(image, target_width=800):
    """Resizes an OpenCV image array to a specific width while maintaining aspect ratio."""
    (h, w) = image.shape[:2]
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def apply_focus_peaking(image, threshold=30):
    """Overlays neon green pixels on the sharpest edges in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    _, mask = cv2.threshold(lap, threshold, 255, cv2.THRESH_BINARY)

    peaked_image = image.copy()
    peaked_image[mask == 255] = [0, 255, 0]  # BGR
    return peaked_image

def main():
    with EasyBasler() as cam:
        if not cam._check_open():
            print("Failed to open camera. Exiting.")
            return

        # Use the full sensor. These nodes may be locked or unavailable on some cameras.
        try:
            cam.camera.OffsetX.SetValue(0)
            cam.camera.OffsetY.SetValue(0)
            cam.camera.Width.SetValue(cam.camera.WidthMax.GetValue())
            cam.camera.Height.SetValue(cam.camera.HeightMax.GetValue())
        except Exception:
            pass

        # Freerun
        cam.set_trigger_mode(enable_hardware_trigger=False)

        window_name = 'Basler - Live View & Focus Assist'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        print("\n--- Controls ---")
        print("[ t ] - Toggle Focus Peaking ON / OFF")
        print("[ c ] - Capture & save uncompressed high-res image (.tiff)")
        print("[ w ] - Make peaking LESS sensitive (higher threshold)")
        print("[ s ] - Make peaking MORE sensitive (lower threshold)")
        print("[ q ] - Quit\n")

        peaking_threshold = 30
        show_peaking = True

        cam.start_grabbing()

        try:
            while True:
                raw_img = cam.get_next_frame(timeout_ms=5000)

                if raw_img is None:
                    continue  # Grab timed out or failed

                # raw_img stays at full resolution for saving
                display_img = resize_frame_keep_ratio(raw_img, target_width=1000)

                if show_peaking:
                    final_display = apply_focus_peaking(display_img, threshold=peaking_threshold)
                    text = f"Peaking: ON | Threshold: {peaking_threshold} (w/s)"
                    color = (0, 255, 0)
                else:
                    final_display = display_img.copy()
                    text = "Peaking: OFF"
                    color = (0, 0, 255)

                cv2.putText(final_display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)

                cv2.imshow(window_name, final_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    show_peaking = not show_peaking
                elif key == ord('c'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.tiff"
                    cv2.imwrite(filename, raw_img)
                    print(f"Saved: {filename} (Resolution: {raw_img.shape[1]}x{raw_img.shape[0]})")
                elif key == ord('w'):
                    peaking_threshold = min(250, peaking_threshold + 5)
                elif key == ord('s'):
                    peaking_threshold = max(5, peaking_threshold - 5)

        except KeyboardInterrupt:
            print("Process interrupted by user.")
        except Exception as e:
            print(f"An exception occurred during the live loop: {e}")

        finally:
            # The 'with' block closes the camera itself
            cam.stop_grabbing()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()