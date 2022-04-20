import numpy as np
import cv2
import yaml

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibration chessboard size
chess_width = 6
chess_height = 5

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_width * chess_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_width, 0:chess_height].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# cap = cv.VideoCapture('out.mp4')
cap = cv2.VideoCapture(0)

if(cap.isOpened() is False):
    print('Error opening video stream of file')

cal_counter = 0

while True:
    ret, frame = cap.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray,
                                                   (chess_height,
                                                    chess_width),
                                                   None)

        if found is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,
                                        corners,
                                        (11, 11),
                                        (-1, -1),
                                        criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(frame,
                                      (chess_width, chess_height),
                                      corners2,
                                      found)
            cv2.imshow('frame', frame)

            height, width = gray.shape[:2]
            # Camera Calibration
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                               imgpoints,
                                                               (width, height),
                                                               None,
                                                               None)
            cal_counter += 1
            print('Cam Calibration')

            # Refine Camera Matrix
            ref_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                             dist,
                                                             (width, height),
                                                             1,
                                                             (width, height))
            print('Cam Matrix Refined')

            # Transform the matrix & distortion coefficients to writable lists
            data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
                    'dist_coeff': np.asarray(dist).tolist(),
                    'rvecs': np.asarray(rvecs).tolist(),
                    'tvecs': np.asarray(tvecs).tolist()}

            # Save calibration parameters to a yaml file
            with open("calibration_matrix_corners2.yaml", "w") as f:
                yaml.dump(data, f)

        if cv2.waitKey(10) == 27:
            break
    else:
        break

cv2.destroyAllWindows()
print('Destroyed all windows')
print(f'Cal_Counter: {cal_counter}')
