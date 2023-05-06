import datetime
import time
from collections import OrderedDict

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from scipy.spatial import distance as dist

from packages.tracking_bounding_boxes import CentroidTracker
confidence = 0.5
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)
# load our serialized model from disk
print("[INFO] loading model...")
model_cafe = "D:\Documents/tracking_face\models/res10_300x300_ssd_iter_140000.caffemodel"
protoxt_model = "D:\Documents/tracking_face\models\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(protoxt_model, model_cafe)
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
src = r"rtsp://admin:space123@192.168.1.49:554"
# src = 0
vs = VideoStream(src=src).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    t1 = time.time()
    # read the next frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)
    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))
            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
    # print(type(rects))

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects, bbx = ct.update(rects)
    # loop over the tracked objects
    for bb in bbx:
        cv2.imwrite(f"images/face/unknown{datetime.datetime.now().strftime('%m%d%Y%H%M%S%F')}.jpg", frame)
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    print(time.time() - t1)
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
