import sys
import socket
import numpy as np
import threading


def get_car_center_str(data):
    if 1 in data and 6 in data:
        car_center = (data[1] + data[6]) / 2
        data_str = f"{car_center[0]},{car_center[1]}"
    elif 2 in data and 5 in data:
        car_center = (data[2] + data[5]) / 2
        data_str = f"{car_center[0]},{car_center[1]}"
    else:
        data_str = "0"

    return data_str


def process_req(conn, addr, data):
    with conn:
        print('Connected by', addr)
        while True:
            data_recv = conn.recv(1024)
            if not data_recv:
                break
            data_split = data_recv.decode("utf8").split(",")
            id = int(data_split[0])
            position = np.array([data_split[1], data_split[2]],
                                dtype=np.float32)
            data[id] = position

            center_str = get_car_center_str(data)
            conn.sendall(bytes(center_str, "utf8"))


def car_center_socket_server(host, port):
    data = {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            sub_threading = threading.Thread(target=process_req,
                                             args=(conn, addr, data),
                                             daemon=True)
            sub_threading.start()


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        id = int(args[1])
        host = args[2]
        port = int(args[3])
    except:
        raise RuntimeError("input args is invalid")
    else:
        if id == 1:
            car_center_socket_server(host, port)

