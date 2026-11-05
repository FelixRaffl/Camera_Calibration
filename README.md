# Camera Calibration

Checkerboard calibration for a Basler camera with OpenCV.

## Setup

```
pip install -r requirements.txt
```

`main.py` also needs the `basler_easy_connect` wrapper, which is not part of this repository.

## Workflow

1. `python main.py`: live view with focus peaking to focus the camera. Press `c` to save a full-resolution TIFF.
2. `python takepics.py`: saves checkerboard images to `calib_images/`. Press `c` to capture while the board is detected, `q` to quit.
3. `Camera_Calibration.ipynb`: calculates the calibration, saves it to `basler_calib_data.npz`, and shows the result on a sample image and on the live camera.

Set `CHECKERBOARD` and `SQUARE_SIZE` to match your board. `takepics.py` and the notebook must use the same `CHECKERBOARD` value.

## Current calibration

| Setting | Value |
| --- | --- |
| Inner corners | 9 x 7 |
| Square size | 10 mm |
| Images | 44 |
| Image size | 2592 x 1944 px |
| Reprojection error | 0.86 px |
