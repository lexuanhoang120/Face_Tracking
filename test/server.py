import socket
import threading
import time

import numpy as np

import cv2


HOST = "0.0.0.0"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

def bytes_to_numpy(bts):
    """
    :param bts: results from image_to_bts
    """
    buffer = np.frombuffer(bts, np.uint8).reshape(1, -1)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
def hello():
    while True:
        print("11")
        time.sleep(3)
    # return 0
threading.Thread(target=hello, args=()).start()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    with conn:
        try:
            while True:

                re = conn.recv(10000000)
                print(re)
                # # img = bytes_to_numpy(re)
                # # print(img.shape)
                # conn.send(b"OK")
                # # cv2.imshow("img",img)
                # # cv2.waitKey(1)
                # re = conn.recv(10000000)
                # img = bytes_to_numpy(re)
                # # cv2.imshow("img", img)
                # # cv2.waitKey(1)
                # print(img.shape)
                # conn.send(b"ggggggg")
                # print("senmt")
                # break
        except:
            print("done")


