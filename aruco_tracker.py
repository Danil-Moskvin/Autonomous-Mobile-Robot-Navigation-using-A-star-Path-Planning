import cv2
import numpy as np


class ArucoTracker:
    def __init__(self, camera_id=1, robot_marker_id=0):
        self.robot_marker_id = robot_marker_id

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Камера не найдена")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.params
        )

    def get_pose(self):
        ret, frame = self.cap.read()

        if not ret:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return None, frame

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        ids_flat = ids.flatten()
        robot_index = None

        for i, marker_id in enumerate(ids_flat):
            if marker_id == self.robot_marker_id:
                robot_index = i
                break

        if robot_index is None:
            cv2.putText(
                frame,
                f"Robot marker ID={self.robot_marker_id} not found",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return None, frame

        marker_corners = corners[robot_index][0]

        center_x = int(marker_corners[:, 0].mean())
        center_y = int(marker_corners[:, 1].mean())

        dx = marker_corners[1][0] - marker_corners[0][0]
        dy = marker_corners[1][1] - marker_corners[0][1]

        angle = np.arctan2(dy, dx)

        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"ROBOT ID={self.robot_marker_id}",
            (center_x + 10, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        pose = {
            "x": center_x,
            "y": center_y,
            "angle": angle,
            "id": int(self.robot_marker_id)
        }

        return pose, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()