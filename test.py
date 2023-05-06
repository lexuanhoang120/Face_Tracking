import socket
import time

HOST = "localhost"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
client_side = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_side.connect((HOST, PORT))
# st = "dsadasd"

while True:
    client_side.send(b"send")
    input("nhap di thang ngu")
    # print("1")
    # time.sleep(5)