# line_detection.py

import cv2
import numpy as np
import time

def get_points_on_line(rho, theta, num_points=20):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(x1 * (1 - t) + x2 * t)
        y = int(y1 * (1 - t) + y2 * t)
        points.append((x, y))
    return points

def detect_lines_in_video(video_path, seconds_inactive=3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error al abrir el video")
        return None, None  # Devolver None si no se puede abrir el video

    last_left_line = None
    last_right_line = None
    last_update_time = None
    persistent_left_line = None
    persistent_right_line = None

    line1_coordinates = []
    line2_coordinates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        left_line = None
        right_line = None
        min_rho = float('inf')
        max_rho = float('-inf')
        target_theta = None

        if lines is not None:
            for rho, theta in lines[:, 0]:
                if target_theta is None:
                    target_theta = theta

                angle_tolerance = np.pi / 5
                if abs(theta - target_theta) <= angle_tolerance:
                    if rho < min_rho:
                        min_rho = rho
                        left_line = (rho, theta)
                    if rho > max_rho:
                        max_rho = rho
                        right_line = (rho, theta)

        current_time = time.time()

        if left_line == last_left_line and right_line == last_right_line:
            if last_update_time is None or current_time - last_update_time >= seconds_inactive:
                persistent_left_line = left_line
                persistent_right_line = right_line
                # Si las líneas permanecen estáticas por el tiempo requerido, terminamos el proceso
                print("Líneas detectadas y quietas por el tiempo necesario. Cerrando el video.")
                cap.release()
                cv2.destroyAllWindows()
                return get_points_on_line(*persistent_left_line), get_points_on_line(*persistent_right_line)
        else:
            last_update_time = current_time

        last_left_line = left_line
        last_right_line = right_line

        if persistent_left_line is not None:
            rho, theta = persistent_left_line
            line1_coordinates = get_points_on_line(rho, theta)

        if persistent_right_line is not None:
            rho, theta = persistent_right_line
            line2_coordinates = get_points_on_line(rho, theta)

        cv2.imshow('Video - Lineas detectadas y puntos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return line1_coordinates, line2_coordinates

