import apriltag
import numpy as np
import cv2
import sys
import argparse
import time
from imutils.video import VideoStream
import yaml

def DrawMarkers(frame, corners, ids):

	# Verify *at least* one ArUco marker was detected
	if len(corners) > 0:

		# Flatten the ArUco IDs list
		# ids = ids.flatten()

	# 	# Loop over the detected ArUCo corners
	# for (markerCorner, markerID) in zip(corners, ids):

		''' Extract the marker corners (which are always returned
		in top-left, top-right, bottom-right, and bottom-left
		order)'''
		# corners = markerCorner.reshape((4, 2))
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
		cv2.putText(frame, str(ids),
			(topLeft[0], topLeft[1] - 15),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)

def EstimatePose(frame, ids, camera_matrix,dst, corners):
	if np.all(ids is not None):  # If there are markers found by detector
		# Estimate Pos of all markers
		# for i in range(len(ids)):  # Iterate in markers
		marker_size = 0.0635 # meters
		rvecs_markers, tvecs_markers, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dst)
		cv2.aruco.drawAxis(frame, camera_matrix, dst, rvecs_markers, tvecs_markers, 0.05)  # Draw Axis
		print(f'Rotation Vectors: {rvecs_markers}')
		print(f'Translation Vectors: {tvecs_markers}')

if __name__ == '__main__':
		
	# Read YAML file fro Camera Calibration info
	with open("calibration_matrix.yaml", 'r') as stream:
		data_loaded = yaml.safe_load(stream)

	camera_matrix = np.asarray(data_loaded['camera_matrix']) # Camera Matrix
	dst = np.asarray(data_loaded['dist_coeff']) # Distortion coefficients
	rvecs = np.asarray(data_loaded['rvecs']) # Rotation Vectors
	tvecs = np.asarray(data_loaded['tvecs']) # Translation Vectors


	# Construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--type", type=str,
		default="tag36h11",
		help="type of AprilTag tag to detect")
	args = vars(ap.parse_args())

	# Initialize AprilTag type for detection
	# detector = apriltag(args["type"])
	options = apriltag.DetectorOptions(families=args["type"])
	detector = apriltag.Detector(options)
	
	# Initialize the video stream and allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

	while True:
		# Get frame from threaded video stream
		frame = vs.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect April Tags
		tags = detector.detect(gray)

		# Draw outline and tag ID for all Tags detected
		for tag in tags:
			DrawMarkers(frame, tag.corners, tag.tag_id)
			EstimatePose(frame, tag.tag_id, camera_matrix, dst, tag.corners) # Draw axis and print pose estimate of detected markers


		# Show output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		# If `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# Close windows and end video stream
	cv2.destroyAllWindows()
	vs.stop()