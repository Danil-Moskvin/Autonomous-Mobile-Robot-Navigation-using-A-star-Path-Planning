import socket
import struct
import numpy as np


class RobotController:
    def __init__(self, ip="192.168.1.158", port=4210, max_output=90):
        self.ip = ip
        self.port = port
        self.max_output = max_output
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_motors(self, values):
        values = np.clip(values, -255, 255).astype(np.int16)
        data = struct.pack("<4h", *values)
        self.sock.sendto(data, (self.ip, self.port))

    def move(self, forward, strafe):
        motor_fl = forward + strafe
        motor_fr = forward - strafe
        motor_bl = forward - strafe
        motor_br = forward + strafe

        motors = np.array([
            motor_fr,
            motor_fl,
            motor_bl,
            motor_br
        ])

        max_abs = np.max(np.abs(motors))

        if max_abs > 1:
            motors = motors / max_abs

        motors = motors * self.max_output
        self.send_motors(motors)

    def stop(self):
        self.send_motors(np.array([0, 0, 0, 0]))

    def close(self):
        self.stop()
        self.sock.close()