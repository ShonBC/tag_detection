import numpy as np
from numpy import linalg
import cv2
import yaml


def ReprojectionError(imgpoints, objpoints, mtx, dist, rvecs, tvecs):

    total_err = 0
    # view_err.resize(objpoints.size())

    for i in range(len(objpoints)):
        new_points, _ = cv2.projectPoints(objpoints[i],
                                          rvecs[i],
                                          tvecs[i],
                                          mtx,
                                          dist)

        err = cv2.norm(imgpoints[i], new_points, cv2.NORM_L2) / len(new_points)

        total_err += err

    return np.sqrt(total_err / len(objpoints))


def Calibrate():
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Calibration chessboard size
    chess_width = 9
    chess_height = 6

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

    view_err = 0

    while True:
        ret, frame = cap.read()

        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(gray,
                                                       (chess_height,
                                                        chess_width),
                                                       None)

            if found is True:
                cal_counter += 1
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
                cv2.imwrite(f'cal_{cal_counter}.jpg', frame)

                height, width = gray.shape[:2]
                # Camera Calibration
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                   imgpoints,
                                                                   (width,
                                                                    height),
                                                                   None,
                                                                   None)

                print(f'Cal Count: {cal_counter}')
                print('Cam Calibration')

                # Refine Camera Matrix
                ref_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                                 dist,
                                                                 (width,
                                                                  height),
                                                                 1,
                                                                 (width,
                                                                  height))
                print('Cam Matrix Refined')

                view_err = ReprojectionError(imgpoints, objpoints, mtx, dist, rvecs, tvecs)

                # Transform matrix & distortion coefficients to writable lists
                # data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
                #         'dist_coeff': np.asarray(dist).tolist(),
                #         'rvecs': np.asarray(rvecs).tolist(),
                #         'tvecs': np.asarray(tvecs).tolist()}

                data = {'camera_matrix': np.asarray(ref_cam_mtx).tolist(),
                        'dist_coeff': np.asarray(dist).tolist()}

                # Save calibration parameters to a yaml file
                with open("calibration_matrix_corners2.yaml", "w") as f:
                    yaml.dump(data, f)
                print(f'Calibrating Error: {view_err}')
                if view_err < 0.90:
                    print(f'Done Calibrating Error: {view_err}')
                    break

            else:
                cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            # If `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        else:
            break

    cv2.destroyAllWindows()
    print('Destroyed all windows')
    print(f'Cal_Counter: {cal_counter}')
    print(f'Calibrating Error: {view_err}')


if __name__ == '__main__':
    Calibrate()
