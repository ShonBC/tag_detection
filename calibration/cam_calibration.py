import glob
import numpy as np
import cv2 as cv
import yaml


def ReprojectionError(imgpoints, objpoints, mtx, dist, rvecs, tvecs):

    total_err = 0
    # view_err.resize(objpoints.size())

    for i in range(len(objpoints)):
        new_points, _ = cv.projectPoints(objpoints[i],
                                         rvecs[i],
                                         tvecs[i],
                                         mtx,
                                         dist)

        err = cv.norm(imgpoints[i], new_points, cv.NORM_L2) / len(new_points)

        total_err += err

    return np.sqrt(total_err / len(objpoints))


# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibration chessboard size
chess_width = 7
chess_height = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chess_width * chess_height, 3), np.float32)
objp[:, :2] = np.mgrid[0:chess_width, 0:chess_height].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


images = glob.glob('calibration/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    found, corners = cv.findChessboardCorners(gray, (chess_height, chess_width), None)

    # If found, add object points, image points (after refining them)
    if found:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img,
                                 (chess_width, chess_height),
                                 corners2,
                                 found)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Camera Calibration
height, width = gray.shape[:2]
# Camera Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,
                                                  imgpoints,
                                                  (width,
                                                   height),
                                                  None,
                                                  None)
# Refine Camera Matrix
img = cv.imread(images[0])
ref_cam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

# Transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),
        'rvecs': np.asarray(rvecs).tolist(),
        'tvecs': np.asarray(tvecs).tolist()}

# Save calibration parameters to a yaml file
with open("calibration_matrix3.yaml", "w") as f:
    yaml.dump(data, f)

view_err = ReprojectionError(imgpoints, objpoints, mtx, dist, rvecs, tvecs)
print(f'Reprojection Err: {view_err}')
