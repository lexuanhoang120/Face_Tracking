import configparser
import multiprocessing
import time

import cv2
from imutils.video.videostream import VideoStream

from packages.alert import Alert
from packages.detecting_faces import detecting_face_from_pillow_image, draw_bounding_boxes
from packages.socket_send_receive import SocketReceiveSend
from packages.tracking_bounding_boxes import CentroidTracker

config = configparser.RawConfigParser()
config.read('conf.txt')


def initiating_parametters(item):
    detail_parameters = dict(config.items(item))
    PADDING = eval(detail_parameters['padding'])
    THRESHOLD_DISTANCE = eval(detail_parameters['threshold_distance'])
    SIZE_SCALE = eval(detail_parameters['size_scale'])
    MAX_DISAPPEARED = eval(detail_parameters['max_disappeared'])
    CONFIDENCE = eval(detail_parameters['confidence'])
    FRAME_TRACKED = eval(detail_parameters['frame_tracked'])
    HOST_SERVER = eval(detail_parameters['host_server'])
    PORT_SERVER = eval(detail_parameters['port_server'])
    try:
        X1 = eval(detail_parameters['x1'])
        X2 = eval(detail_parameters['x2'])
        Y1 = eval(detail_parameters['y1'])
        Y2 = eval(detail_parameters['y2'])
    except:
        X1 = eval("-1")
        X2 = eval("10000")
        Y1 = eval("-1")
        Y2 = eval("10000")
    try:
        SOURCE = eval(detail_parameters['source'])
    except:
        SOURCE = detail_parameters['source']
    SHOW = eval(detail_parameters['show'])
    return SOURCE, PADDING, THRESHOLD_DISTANCE, SIZE_SCALE, MAX_DISAPPEARED, CONFIDENCE, FRAME_TRACKED, X1, X2, Y1, Y2, SHOW, HOST_SERVER,PORT_SERVER


def tracking_camera(item):
    print(f"_____________-Starting tracking {item}___________")
    print(f"-------------------Start reading parameters {item}-------------------------")
    source, padding, threshold_distance, size_scale, max_disappeared, confidence, frame_tracked, x1, x2, y1, y2, is_showed ,host_server, port_server= initiating_parametters(
        item=item)
    print(f"_____________ Start initiating alert ___________")
    alert = Alert()
    print(f"_____________ Start initiating client socket  ___________")
    client = SocketReceiveSend( host=host_server,port=port_server)
    print(f"--------------------------Start reading frame {item}---------------------")
    connection = VideoStream(source).start()
    tracker = CentroidTracker(
        threshold_distance=threshold_distance,
        max_disappeared=max_disappeared,
        confidence=confidence,
        frame_tracked=frame_tracked,
    )

    def receive_alert():
        dts = set()
        while True:
            dt = client.receive_data()
            dts.add(dt)
            for data in dts.copy():
                alert.alert(data)
                dts.remove(data)

    # thread_receive_alert = threading.Thread(target=receive_alert, args=())
    # thread_receive_alert.start()
    while True:
        # error for dont connect to host and port
        print(f"Read frame: {item}")
        frame = connection.read()
        # check signal from connection
        if frame is None:
            print(f"Disconnected: {item}")
            connection.stop()
            time.sleep(5)
            connection = VideoStream(source).start()
            continue
        frame = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]
        # do face tracking and ignore facial landmarks
        bounding_boxes, scores = detecting_face_from_pillow_image(image=frame, size_scale=size_scale)
        # update face tracker
        objects, id_bounding_boxes_face_tracked = tracker.update(bounding_boxes, scores)
        for id_face in id_bounding_boxes_face_tracked:
            startX, startY, endX, endY = bounding_boxes[id_face]
            image = frame[max(startY - padding, 0):min(endY + padding, frame.shape[0]),
                    max(startX - padding, 0):min(endX + padding, frame.shape[1])]

            # error for disconnect server socket
            try:
                client.send_data(image)
            except Exception as e:
                print("Disconnect server socket :", e)
                client = SocketReceiveSend()
                client.send_data(image)


            # a = client.receive_data()
            # client.send_data(frame)
            # t1 = threading.Thread(target=alert.alert, args=(client.receive_data(),)).start()
            # t1.join()
            # alert.alert(dt)

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
            # frame = imutils.resize(frame, width=1000)
            cv2.imshow(f"Camera: {item}", frame)
            tap_key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if tap_key == ord("q"):
                cv2.destroyAllWindows()
                break

        else:
            continue
    client.client_side.close()
    connection.stop()


if __name__ == "__main__":

    processes = []

    for key, value in dict(config.items()).items():
        if key == "DEFAULT":
            continue
        else:
            print(f"------------------Add {key} to process---------------------")
            processes.append(multiprocessing.Process(target=tracking_camera, args=(key,)))
    for process in processes:
        print(f"------------------Start processing---------------------")
        process.start()
