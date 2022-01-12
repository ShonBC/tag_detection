# Dependencies:
- Ubuntu 18.04
- Python - 3.8.12
- OpenCV - 4.5.4

# Install Dependencies
A bash script was made to easily install all necessary dependencies and packages.

    ./dep.sh

# Camera Calibration
Take a minimum of 10 images for camera calibration. Run take_pic.py to take a picture. 

    python3 take_pic.py

Rename and save the image to the calibration directory. 

- ex: cal_1.jpg, cal_2.jpg... 

Run the camera calibration script to generate a yaml file containing the camera matrix, distortion coefficients, as well as the rotation and translation vectors.

    python3 cam_calibration.py

# ArUco
ArUco tags have many useful applications. More information can be found [here](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html). This module was designed to detect, decode, and estimate the position and orientation of ArUco Tags seen be a camera using OpenCV. The tags being used should be measured and their size updated in the code. The EstimatePose() function has a variable "marker_size" which should be updated to match the actual length of the one of the tag's edges in meters. ArUco Markers can be generated [here](https://chev.me/arucogen/). Detect the default ArUco Tag type by running:

    python aruco_pos.py

Change tag type by running:

    python aruco_pos.py -t {TAG_TYPE}

For example:

    python aruco_pos.py -t DICT_4X4_50