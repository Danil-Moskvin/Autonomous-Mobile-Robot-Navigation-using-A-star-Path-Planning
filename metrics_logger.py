import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MetricsLogger:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.rows = []

    def add(
        self,
        t,
        x,
        y,
        goal_error,
        target_error,
        path_error,
        forward,
        strafe,
        speed_px_s
    ):
        self.rows.append({
            "time": float(t),
            "x": float(x),
            "y": float(y),
            "goal_error": float(goal_error),
            "target_error": float(target_error),
            "path_error": float(path_error),
            "forward": float(forward),
            "strafe": float(strafe),
            "speed_px_s": float(speed_px_s)
        })

    def save_csv(self, filename="metrics.csv"):
        if len(self.rows) == 0:
            return

        path = os.path.join(self.results_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)

        print("CSV сохранен:", path)

    def plot_all(self):
        if len(self.rows) == 0:
            print("Нет данных для построения графиков")
            return

        t = np.array([r["time"] for r in self.rows])
        x = np.array([r["x"] for r in self.rows])
        y = np.array([r["y"] for r in self.rows])
        goal_error = np.array([r["goal_error"] for r in self.rows])
        target_error = np.array([r["target_error"] for r in self.rows])
        path_error = np.array([r["path_error"] for r in self.rows])
        forward = np.array([r["forward"] for r in self.rows])
        strafe = np.array([r["strafe"] for r in self.rows])
        speed = np.array([r["speed_px_s"] for r in self.rows])

        self._plot_line(
            t,
            goal_error,
            "Ошибка до цели",
            "Время, с",
            "Ошибка, пиксели",
            "goal_error.png"
        )

        self._plot_line(
            t,
            target_error,
            "Ошибка до текущей точки пути",
            "Время, с",
            "Ошибка, пиксели",
            "target_error.png"
        )

        self._plot_line(
            t,
            path_error,
            "Отклонение от построенного пути",
            "Время, с",
            "Отклонение, пиксели",
            "path_error.png"
        )

        plt.figure()
        plt.plot(t, forward, label="forward")
        plt.plot(t, strafe, label="strafe")
        plt.xlabel("Время, с")
        plt.ylabel("Команда управления")
        plt.title("Команды управления роботом")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, "control_commands.png"), dpi=200)
        plt.close()

        self._plot_line(
            t,
            speed,
            "Скорость робота по изображению",
            "Время, с",
            "Скорость, пикс/с",
            "speed.png"
        )

        plt.figure()
        plt.plot(x, y)
        plt.gca().invert_yaxis()
        plt.xlabel("x, пиксели")
        plt.ylabel("y, пиксели")
        plt.title("Реальная траектория робота")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "robot_trajectory.png"), dpi=200)
        plt.close()

        print("Графики сохранены в папку:", self.results_dir)

    def save_trajectory_on_map(self, map_img, filename="trajectory_on_map.png"):
        if len(self.rows) == 0:
            return

        img = map_img.copy()

        points = [
            (int(r["x"]), int(r["y"]))
            for r in self.rows
        ]

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (255, 255, 0), 2)

        for p in points:
            cv2.circle(img, p, 2, (255, 255, 0), -1)

        path = os.path.join(self.results_dir, filename)
        cv2.imwrite(path, img)

        print("Траектория на карте сохранена:", path)

    def _plot_line(self, x, y, title, xlabel, ylabel, filename):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, filename), dpi=200)
        plt.close()