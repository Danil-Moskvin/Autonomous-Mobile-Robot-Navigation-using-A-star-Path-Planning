import cv2
import numpy as np
import heapq


def load_map(path, size=(1280, 720), obstacle_dilation=15):
    img = cv2.imread(path)

    if img is None:
        raise RuntimeError("Карта не найдена")

    img = cv2.resize(img, size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 80, 80])
    upper_red_2 = np.array([180, 255, 255])

    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    obstacles = cv2.bitwise_or(red_mask_1, red_mask_2)

    if obstacle_dilation > 0:
        kernel = np.ones((obstacle_dilation, obstacle_dilation), np.uint8)
        obstacles = cv2.dilate(obstacles, kernel, iterations=1)

    black_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 80])
    )

    free_space = (black_mask > 0) & (obstacles == 0)

    return free_space, img, obstacles


def find_goal(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    points = cv2.findNonZero(green_mask)

    if points is None:
        raise RuntimeError("Зеленая цель не найдена на карте")

    center = points.mean(axis=0)[0]

    return int(center[0]), int(center[1])


def find_nearest_free(grid, point, max_radius=200):
    x, y = point
    h, w = grid.shape

    if 0 <= x < w and 0 <= y < h and grid[y, x]:
        return point

    for r in range(1, max_radius):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx = x + dx
                ny = y + dy

                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx]:
                        return nx, ny

    return None


def astar(grid, start, goal, step=10):
    height, width = grid.shape

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [
        (step, 0),
        (-step, 0),
        (0, step),
        (0, -step),
        (step, step),
        (step, -step),
        (-step, step),
        (-step, -step)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)

        if heuristic(current, goal) < step:
            path = [goal]

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)
            path.reverse()

            return path, visited

        for dx, dy in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy

            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue

            if not grid[ny, nx]:
                continue

            neighbor = (nx, ny)
            cost = np.hypot(dx, dy)
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, visited


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0

    length = 0.0

    for i in range(len(path) - 1):
        length += np.hypot(
            path[i + 1][0] - path[i][0],
            path[i + 1][1] - path[i][1]
        )

    return length


def distance_to_path(point, path):
    if path is None or len(path) == 0:
        return 0.0

    x, y = point
    distances = [
        np.hypot(x - px, y - py)
        for px, py in path
    ]

    return float(min(distances))


def draw_debug_map(img, obstacles, path=None, visited=None, start=None, goal=None):
    debug = img.copy()

    obstacle_overlay = np.zeros_like(debug)
    obstacle_overlay[obstacles > 0] = (0, 0, 255)

    debug = cv2.addWeighted(debug, 0.7, obstacle_overlay, 0.3, 0)

    if visited is not None:
        for p in visited:
            cv2.circle(debug, p, 1, (255, 255, 0), -1)

    if path is not None:
        for i in range(len(path) - 1):
            cv2.line(debug, path[i], path[i + 1], (255, 0, 0), 3)

        for p in path:
            cv2.circle(debug, p, 3, (255, 0, 0), -1)

    if start is not None:
        cv2.circle(debug, start, 10, (0, 255, 255), -1)
        cv2.putText(
            debug,
            "START",
            (start[0] + 10, start[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    if goal is not None:
        cv2.circle(debug, goal, 10, (0, 255, 0), -1)
        cv2.putText(
            debug,
            "GOAL",
            (goal[0] + 10, goal[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return debug
