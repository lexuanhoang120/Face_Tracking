import time

import imutils
import cv2
from imutils.video.videostream import VideoStream

from packages.detecting_faces import detecting_face_from_pillow_image, locating_centroids
from test.drawing import draw_tracker, draw_bounding_boxes
from packages.tracking_bounding_boxes import CentroidTracker


def display(frame, tracker, bboxes, scores):
    canvas = frame.copy()
    canvas = draw_tracker(canvas, tracker)
    canvas = draw_bounding_boxes(canvas, bboxes, scores)
    return canvas


def track_video(camera_id):

    vcap = VideoStream(camera_id).start()

    tracker = CentroidTracker()
    while True:
        t1 = time.time()
        frame = vcap.read()
        frame = imutils.resize(frame, width=1000)

        # do face tracking and ignore facial landmarks
        bboxes, scores = detecting_face_from_pillow_image(frame)
        # update face tracker
        centroids = locating_centroids(bboxes)
        tracker.update(centroids)
        # cv2.waitKey(1)
        # print(tracker.objects)
        # break
        # show results
        if len(centroids) > 0:
            frame = display(frame, tracker, bboxes, scores)

        cv2.imshow("fds", frame)
        cv2.waitKey(1)
        print(t1 - time.time())
    # cv2.destroyAllWindows()
    # vcap.release()


src = r"rtsp://admin:space123@192.168.1.49:554"
# src = 0
track_video(src)
