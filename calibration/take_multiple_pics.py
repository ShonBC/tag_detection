'''
Take 10 pictures for camera calibration. Images are saved as "cal_#.jpg.
'''

import cv2

# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera

i = 0

while True:

    # Get frame from threaded video streamx
    ret, frame = cam.read()
    
    if not ret or i >= 10:
        break

    # Show output frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):    # frame captured without any errors
        cv2.imshow("cam-test",frame)
        cv2.waitKey(0)
        cv2.destroyWindow("cam-test")
        cv2.imwrite(f"cal_{i}.jpg",frame) #save image
        i = i + 1
    