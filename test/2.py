import cv2
from imutils.video.videostream import VideoStream

src  = r"rtsp://admin:sp@ce123@192.168.1.61:554"
cap = VideoStream(src= src ).start()
while True:
    frame = cap.read()
    x1 = 200
    x2 = 1500
    y1 = 0
    y2 = 900
    frame = frame[y1:y2,x1:x2]
    cv2.imshow("camera",frame)
    cv2.waitKey(1)