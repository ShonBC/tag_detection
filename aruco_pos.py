import numpy as np
import cv2
import sys
import argparse
import time
from imutils.video import VideoStream
import yaml
from scipy.spatial.transform import Rotation as R

def DrawMarkers(frame, corners, ids):

	# Verify *at least* one ArUco marker was detected
	if len(corners) > 0:

		# Flatten the ArUco IDs list
		ids = ids.flatten()

		# Loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):

			''' Extract the marker corners (which are always returned
			in top-left, top-right, bottom-right, and bottom-left
			order)'''
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# Convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			''' Compute and draw the center (x, y)-coordinates of the
			ArUco marker'''
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
			
			# Draw the ArUco marker ID on the frame
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

def EstimatePose(frame, ids, camera_matrix,dst, corners):
	if np.all(ids is not None):  # If there are markers found by detector
		# Estimate Pos of all markers
		for i in range(len(ids)):  # Iterate in markers
			marker_size = 0.0635 # meters
			# marker_size = 2.5 # inches
			rvecs_markers, tvecs_markers, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, dst)
			cv2.aruco.drawAxis(frame, camera_matrix, dst, rvecs_markers, tvecs_markers, 0.05)  # Draw Axis
			r_mat = cv2.Rodrigues(rvecs_markers)[0]
			r = R.from_matrix(r_mat)
			quat = r.as_quat()
			print(f'Rotation Vectors: {rvecs_markers}')
			print(f'Rotation Matrix: {r_mat}')
			print(f'Quaternion: {quat}')
			print(f'Translation Vectors: {tvecs_markers}')

if __name__ == '__main__':
		
	# Read YAML file fro Camera Calibration info
	with open("calibration_matrix_test.yaml", 'r') as stream:
		data_loaded = yaml.safe_load(stream)

	camera_matrix = np.asarray(data_loaded['camera_matrix']) # Camera Matrix
	dst = np.asarray(data_loaded['dist_coeff']) # Distortion coefficients
	rvecs = np.asarray(data_loaded['rvecs']) # Rotation Vectors
	tvecs = np.asarray(data_loaded['tvecs']) # Translation Vectors


	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--type", type=str,
		default="DICT_ARUCO_ORIGINAL",
		help="type of ArUCo tag to detect")
	args = vars(ap.parse_args())

	# Define names possible ArUco tags OpenCV supports
	ARUCO_DICT = {
		"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
		"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
		"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
		"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
		"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
		"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
		"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
		"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
		"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
		"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
		"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
		"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
		"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
		"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
		"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
		"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
		"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
		"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
		"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
		"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
		"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
	}

	''' Verify that the supplied ArUCo tag exists and is supported by
	OpenCV'''
	if ARUCO_DICT.get(args["type"], None) is None:
		print("[INFO] ArUCo tag of '{}' is not supported".format(
			args["type"]))
		sys.exit(0)

	# Load the ArUCo dictionary, grab the ArUCo parameters
	print("[INFO] detecting '{}' tags...".format(args["type"]))
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
	arucoParams = cv2.aruco.DetectorParameters_create()

	# Initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	# vs = VideoStream(src=0).start()
	cap = cv2.VideoCapture(0)
	time.sleep(2.0)

	while True:
		# Get frame from threaded video stream
		# frame = vs.read()
		ret, frame = cap.read()

		# Detect ArUco markers
		(corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
			arucoDict, parameters=arucoParams)
		
		DrawMarkers(frame, corners, ids) # Draw and label detected markers
		EstimatePose(frame, ids, camera_matrix, dst, corners) # Draw axis and print pose estimate of detected markers

		# Show output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# If `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# Close windows and end video stream
	cv2.destroyAllWindows()
	# vs.stop()
	cap.release()