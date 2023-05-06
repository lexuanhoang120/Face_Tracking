import cv2
from imutils.video.videostream import VideoStream
import time
import imutils
from packages.detecting_faces import *
src = "rtsp://admin:sp@ce123@118.69.224.67:5543"
connection = VideoStream(src).start()
while True:
    frame = connection.read()
    if frame is None:
        connection.stop()
        src = "rtsp://admin:sp@ce123@118.69.224.67:5543"
        connection = VideoStream(src).start()
        time.sleep(2)
        print("disconnected")
        continue
    bboxs,scores = detecting_face_from_pillow_image(frame)
    frame = draw_bounding_boxes(frame,bboxs,scores)
    frame = imutils.resize(frame,width=1000)
    cv2.imshow('test_camera',frame)
    cv2.waitKey(1)

