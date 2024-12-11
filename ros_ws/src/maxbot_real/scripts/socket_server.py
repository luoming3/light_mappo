import socket
import threading
import numpy as np
from status import *

master_status = STATUS_STOP
id_position = {}
id_status = {}

def process_data(id, status):
    global all_status, master_status

    id_status[id] = status
    if status == STATUS_RUNNING:
        pass
    elif status == STATUS_SUCCESS:
        master_status = status
    elif status == STATUS_FAILURE:
        master_status = status
    elif status == STATUS_TURN:
        master_status = status
    elif status == STATUS_STOP:
        # if all status of maxbot are STATUS_STOP, then run it
        if sum(id_status.values()) % STATUS_STOP == 0:
            master_status = STATUS_RUNNING
    else:
        raise RuntimeError("unknown status")

    if 1 in id_position and 6 in id_position:
        car_center = (id_position[1] + id_position[6]) / 2
        data_str = f"1,{car_center[0]},{car_center[1]},{master_status}"
    elif 2 in id_position and 5 in id_position:
        car_center = (id_position[2] + id_position[5]) / 2
        data_str = f"1,{car_center[0]},{car_center[1]},{master_status}"
    else:
        data_str = "0"

    return data_str


def process_req(conn, addr):
    global id_position
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
            id_position[id] = position
            status = int(data_split[3])

            center_str = process_data(id, status)
            conn.sendall(bytes(center_str, "utf8"))


def car_center_socket_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            sub_threading = threading.Thread(target=process_req,
                                            args=(conn, addr),
                                            daemon=True)
            sub_threading.start()

