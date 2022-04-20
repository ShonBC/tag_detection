
import cv2

cap = cv2.VideoCapture(0)

fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 20, (640, 480))

while True:
    ret, frame = cap.read()

    out.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()
