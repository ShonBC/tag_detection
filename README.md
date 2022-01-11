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
