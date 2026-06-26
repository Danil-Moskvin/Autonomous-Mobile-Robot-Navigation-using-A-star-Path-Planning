import cv2
import time
import numpy as np

from aruco_tracker import ArucoTracker
from path_planner import (
    load_map,
    find_goal,
    find_nearest_free,
    astar,
    path_length,
    distance_to_path,
    draw_debug_map
)
from robot_control import RobotController
from metrics_logger import MetricsLogger


MAP_PATH = "experiment/test_3.jpg"

CAMERA_ID = 1
ROBOT_MARKER_ID = 0

ESP32_IP = "192.168.1.158"
ESP32_PORT = 4210

MAX_OUTPUT = 90

TARGET_POINT_OFFSET = 8

GOAL_THRESHOLD = 35
POINT_THRESHOLD = 25

KP_MOVE = 0.006
MAX_MOVE = 0.65

ENABLE_ROBOT_MOVEMENT = True
WAIT_BEFORE_START = True

ROBOT_ANGLE_OFFSET = -np.pi / 2


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def world_vector_to_robot_commands(dx, dy, robot_angle):
    distance = np.hypot(dx, dy)

    if distance < 1e-6:
        return 0.0, 0.0

    vx = dx / distance
    vy = dy / distance

    front_x = np.cos(robot_angle)
    front_y = np.sin(robot_angle)

    right_x = np.cos(robot_angle + np.pi / 2)
    right_y = np.sin(robot_angle + np.pi / 2)

    forward_dir = vx * front_x + vy * front_y
    strafe_dir = vx * right_x + vy * right_y

    move_power = KP_MOVE * distance
    move_power = np.clip(move_power, 0.0, MAX_MOVE)

    forward = move_power * forward_dir
    strafe = move_power * strafe_dir

    forward = np.clip(forward, -MAX_MOVE, MAX_MOVE)
    strafe = np.clip(strafe, -MAX_MOVE, MAX_MOVE)

    return forward, strafe


def main():
    tracker = ArucoTracker(
        camera_id=CAMERA_ID,
        robot_marker_id=ROBOT_MARKER_ID
    )

    robot = RobotController(
        ip=ESP32_IP,
        port=ESP32_PORT,
        max_output=MAX_OUTPUT
    )

    logger = MetricsLogger(results_dir="results")

    free_map, map_img, obstacles = load_map(
        MAP_PATH,
        size=(1280, 720),
        obstacle_dilation=15
    )

    goal = find_goal(map_img)

    print("Цель:", goal)

    path = None
    visited = None
    path_index = 0
    movement_started = False

    start_time = None
    finish_time = None

    previous_position = None
    previous_time = None

    try:
        while True:
            pose, frame = tracker.get_pose()

            if pose is None:
                print("ArUco робота не найден")
                robot.stop()

                if frame is not None:
                    cv2.imshow("Camera", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                continue

            robot_x = pose["x"]
            robot_y = pose["y"]
            robot_angle = normalize_angle(pose["angle"] + ROBOT_ANGLE_OFFSET)

            start = (robot_x, robot_y)

            h, w = free_map.shape

            if not (0 <= start[0] < w and 0 <= start[1] < h):
                print("Старт за пределами карты:", start)
                robot.stop()
                break

            if not free_map[start[1], start[0]]:
                print("Старт попал не в свободную зону:", start)

                new_start = find_nearest_free(
                    free_map,
                    start,
                    max_radius=200
                )

                if new_start is None:
                    print("Не удалось найти свободную точку рядом со стартом")
                    robot.stop()
                    break

                print("Новый старт:", new_start)
                start = new_start

            if not free_map[goal[1], goal[0]]:
                print("Цель попала не в свободную зону:", goal)

                new_goal = find_nearest_free(
                    free_map,
                    goal,
                    max_radius=200
                )

                if new_goal is None:
                    print("Не удалось найти свободную точку рядом с целью")
                    robot.stop()
                    break

                print("Новая цель:", new_goal)
                goal = new_goal

            path, visited = astar(
                free_map,
                start,
                goal,
                step=10
            )

            debug_map = draw_debug_map(
                map_img,
                obstacles,
                path=path,
                visited=visited,
                start=start,
                goal=goal
            )

            if path is None:
                print("Путь не найден")
                print("Старт:", start)
                print("Цель:", goal)
                print("Посещено точек:", len(visited))

                robot.stop()

                cv2.imshow("DEBUG MAP", debug_map)
                cv2.imshow("Camera", frame)
                cv2.waitKey(0)
                break

            if not movement_started and WAIT_BEFORE_START:
                robot.stop()

                cv2.imshow("DEBUG MAP", debug_map)
                cv2.imshow("Camera", frame)
                cv2.waitKey(1)

                print("Путь найден.")
                print("Длина построенного пути:", round(path_length(path), 2), "пикселей")
                print("Проверь карту, старт, цель и путь.")
                time.sleep(5)

                movement_started = True
                start_time = time.time()
                previous_time = start_time
                previous_position = start

            if not movement_started and not WAIT_BEFORE_START:
                movement_started = True
                start_time = time.time()
                previous_time = start_time
                previous_position = start

            current_time = time.time()
            elapsed_time = current_time - start_time

            distance_to_goal = np.hypot(
                goal[0] - start[0],
                goal[1] - start[1]
            )

            if distance_to_goal < GOAL_THRESHOLD:
                finish_time = time.time()
                total_time = finish_time - start_time

                print("Цель достигнута")
                print("Время прохождения пути:", round(total_time, 2), "с")

                robot.stop()
                break

            target_index = min(
                path_index + TARGET_POINT_OFFSET,
                len(path) - 1
            )

            target = path[target_index]

            dx = target[0] - start[0]
            dy = target[1] - start[1]

            distance_to_target = np.hypot(dx, dy)

            if distance_to_target < POINT_THRESHOLD:
                path_index = min(
                    path_index + 1,
                    len(path) - 1
                )

            forward, strafe = world_vector_to_robot_commands(
                dx,
                dy,
                robot_angle
            )

            if previous_position is not None and previous_time is not None:
                dt = current_time - previous_time

                if dt > 1e-6:
                    speed_px_s = np.hypot(
                        start[0] - previous_position[0],
                        start[1] - previous_position[1]
                    ) / dt
                else:
                    speed_px_s = 0.0
            else:
                speed_px_s = 0.0

            previous_position = start
            previous_time = current_time

            path_error = distance_to_path(start, path)

            logger.add(
                t=elapsed_time,
                x=start[0],
                y=start[1],
                goal_error=distance_to_goal,
                target_error=distance_to_target,
                path_error=path_error,
                forward=forward,
                strafe=strafe,
                speed_px_s=speed_px_s
            )

            if ENABLE_ROBOT_MOVEMENT:
                robot.move(
                    forward=forward,
                    strafe=strafe
                )
            else:
                robot.stop()

            cv2.circle(
                frame,
                start,
                8,
                (0, 255, 255),
                -1
            )

            cv2.circle(
                frame,
                goal,
                10,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                f"t={elapsed_time:.1f}s",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            print(
                f"t={elapsed_time:.2f}, "
                f"robot=({robot_x}, {robot_y}), "
                f"target={target}, "
                f"goal_error={distance_to_goal:.1f}, "
                f"target_error={distance_to_target:.1f}, "
                f"path_error={path_error:.1f}, "
                f"forward={forward:.2f}, "
                f"strafe={strafe:.2f}, "
                f"speed={speed_px_s:.1f}px/s"
            )

            cv2.imshow("DEBUG MAP", debug_map)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                robot.stop()
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    finally:
        robot.stop()
        robot.close()
        tracker.release()

        logger.save_csv("metrics.csv")
        logger.plot_all()
        logger.save_trajectory_on_map(map_img)

        print("Работа программы завершена")


if __name__ == "__main__":
    main()
