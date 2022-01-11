import glob
import numpy as np
import cv2 as cv
import yaml

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibration chessboard size
chess_width = 7
chess_height = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibration/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (chess_height, chess_width), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Refine Camera Matrix
img = cv.imread(images[0])
height, width = img.shape[:2]
ref_cam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

# Transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()}

# Save calibration parameters to a yaml file
with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)
