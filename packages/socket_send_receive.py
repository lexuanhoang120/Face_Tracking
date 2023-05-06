import socket
import time

import cv2
import numpy as np


class SocketReceiveSend:
    def __init__(self, host="localhost", port=65432):
        self.HOST = host  # Standard loopback interface address (localhost)
        self.PORT = port  # Port to listen on (non-privileged ports are > 1023)
        self.client_side = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client_side.connect((self.HOST, self.PORT))
                break
            except Exception as e:
                print("Waiting for openning the server :", e)
                print("--------- Sleeping ------")
                time.sleep(5)

    @staticmethod
    def bytes_to_numpy(bts):
        """
        :param bts: results from image_to_bts
        """
        buffer = np.frombuffer(bts, np.uint8).reshape(1, -1)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    @staticmethod
    def numpy_to_bytes(frame):
        """
        :param frame: WxHx3 ndarray
        """
        _, np_byte = cv2.imencode('.jpg', frame)
        return np_byte.tobytes()

    def send_data(self, image):
        bt = self.numpy_to_bytes(image)
        self.client_side.send(bt)
        return 0

    def receive_data(self):
        dt = self.client_side.recv(1000000)
        return dt
