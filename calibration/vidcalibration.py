import numpy as np
import cv2 as cv
import yaml

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibration chessboard size
chess_width = 6
chess_height = 5

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_width * chess_height, 3), np.float32)
objp[:,:2] = np.mgrid[0:chess_width, 0:chess_height].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# cap = cv.VideoCapture('out.mp4')
cap = cv.VideoCapture(0)

if(cap.isOpened() == False):
    print('Error opening video stream of file')

while True:
    ret, frame = cap.read()

    if ret:

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        found, corners = cv.findChessboardCorners(gray, (chess_height, chess_width), None)

        if found == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

            cv.drawChessboardCorners(frame, (chess_width, chess_height), corners2, found)
            cv.imshow('frame', frame)

            height, width = gray.shape[:2]
            # Camera Calibration
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
            print('Cam Calibration')

            # Refine Camera Matrix
            ref_cam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
            print('Cam Matrix Refined')

            # Transform the matrix and distortion coefficients to writable lists
            data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
                    'dist_coeff': np.asarray(dist).tolist(),
                    'rvecs': np.asarray(rvecs).tolist(),
                    'tvecs': np.asarray(tvecs).tolist()}

            # Save calibration parameters to a yaml file
            with open("calibration_matrix_test.yaml", "w") as f:
                yaml.dump(data, f)

        if cv.waitKey(10) == 27:
            break
    else:
        break

cv.destroyAllWindows()
print('Destroyed all windows')
