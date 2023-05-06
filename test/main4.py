import configparser
import datetime
import time

import cv2
from imutils.video.videostream import VideoStream

from packages.detecting_faces import detecting_face_from_pillow_image, draw_bounding_boxes
from packages.tracking_bounding_boxes import CentroidTracker


# PADDING = 10
# THRESHOLD_DISTANCE = 100
# SIZE_SCALE = 2
# MAX_DISAPPEARED = 15
#
# # SOURCE = r"rtsp://admin:space123@192.168.1.49:554"
# SOURCE = 0
# SHOW = True


def tracking_camera(source, is_showed=True):
    connection = VideoStream(source).start()
    tracker = CentroidTracker(threshold_distance=THRESHOLD_DISTANCE, max_disappeared=MAX_DISAPPEARED)
    while True:
        frame = connection.read()
        # check signal from connection
        if frame is None:
            connection.stop()
            time.sleep(0.1)
            connection.start()
            continue
        # do face tracking and ignore facial landmarks
        bounding_boxes, scores = detecting_face_from_pillow_image(image=frame, size_scale=SIZE_SCALE)
        # update face tracker
        objects, id_bounding_boxes_face_tracked = tracker.update(bounding_boxes, scores)
        for id_face in id_bounding_boxes_face_tracked:
            startX, startY, endX, endY = bounding_boxes[id_face]
            saving_time = datetime.datetime.now().strftime('%m%d%Y%H%M%S%f')
            cv2.imwrite(f"images/face/{saving_time}.jpg",
                        frame[max(startY - PADDING, 0):min(endY + PADDING, frame.shape[0]),
                        max(startX - PADDING, 0):min(endX + PADDING, frame.shape[1])])
            cv2.imwrite(f"images/context/{saving_time}.jpg",
                        frame)
        if is_showed:
            frame = draw_bounding_boxes(frame, bounding_boxes, scores)
            for (objectID, centroid) in objects.items():
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0], centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            continue
    connection.stop()


if __name__ == "__main__":
    config = configparser.RawConfigParser()
    config.read('conf.txt')
    details_dict = dict(config.items('CONFIGURATION'))

    PADDING = eval(details_dict['padding'])
    THRESHOLD_DISTANCE = eval(details_dict['threshold_distance'])
    SIZE_SCALE = eval(details_dict['size_scale'])
    MAX_DISAPPEARED = eval(details_dict['max_disappeared'])
    # SOURCE = r"rtsp://admin:space123@192.168.1.49:554"
    try:
        SOURCE = eval(details_dict['source'])
    except:
        SOURCE = details_dict['source']
    SHOW = eval(details_dict['show'])

    tracking_camera(source=SOURCE, is_showed=SHOW)
