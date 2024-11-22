import cv2
import torch
import numpy as np
from sort import Sort
from facenet_pytorch import MTCNN
from PIL import Image

# Verifica si tienes una GPU disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# Inicializa el modelo MTCNN para la detección de rostros
mtcnn = MTCNN(keep_all=True, device=device)

# Inicializar el tracker Sort para el seguimiento de rostros
tracker_face = Sort()

# Coordenadas de las líneas para carril derecho e izquierdo
line1_coordinates = [(84, 217), (129.5, 303.6), (175, 390), (220.5, 476.4), (266, 563), (295, 923)]
line2_coordinates = [(230, 235), (303.8, 299.7), (377.6, 364.3), (451.4, 428.9), (525.2, 493.5), (609, 558), (688.8, 624.1), (709, 930)]

# Función para encontrar el punto más cercano en cada línea
def find_closest_points(center, line1_coords, line2_coords):
    center_x, center_y = center
    closest_point_line1, closest_point_line2 = None, None
    min_y_diff_line1, min_y_diff_line2 = float('inf'), float('inf')

    # Buscar punto más cercano en línea 1
    for point in line1_coords:
        x, y = point
        y_diff = abs(y - center_y)
        if y_diff < min_y_diff_line1:
            min_y_diff_line1 = y_diff
            closest_point_line1 = point

    # Buscar punto más cercano en línea 2
    for point in line2_coords:
        x, y = point
        y_diff = abs(y - center_y)
        if y_diff < min_y_diff_line2:
            min_y_diff_line2 = y_diff
            closest_point_line2 = point

    # Calcular distancia en X para puntos más cercanos
    distance_x_line1 = abs(closest_point_line1[0] - center_x) if closest_point_line1 else float('inf')
    distance_x_line2 = abs(closest_point_line2[0] - center_x) if closest_point_line2 else float('inf')

    return 1 if distance_x_line1 < distance_x_line2 else 2

# Procesar el video
video_path = 'pics/main_prueba.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = 'pics/prueba_result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Variables para almacenar las posiciones anteriores y calcular velocidad
prev_centers = {}
show_bounding_boxes, show_points = True, True

# Bucle para procesar el video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        show_bounding_boxes = not show_bounding_boxes
    if key == ord('n'):
        show_points = not show_points

    # Convertir frame para MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Detectar rostros
    boxes, _ = mtcnn.detect(img_pil)
    detections_for_sort = [[int(box[0]), int(box[1]), int(box[2]), int(box[3]), 1] for box in boxes] if boxes is not None else []

    # Seguimiento de rostros
    tracked_objects = tracker_face.update(np.array(detections_for_sort)) if detections_for_sort else np.empty((0, 5))

    # Dibujar cajas de detección y puntos centrales
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Determinar carril
        closest_line = find_closest_points((center_x, center_y), line1_coordinates, line2_coordinates)
        carril = "Izquierdo" if closest_line == 2 else "Derecho"

        # Calcular velocidad
        if obj_id in prev_centers:
            prev_center = prev_centers[obj_id]
            distance = np.sqrt((center_x - prev_center[0]) ** 2 + (center_y - prev_center[1]) ** 2)
            speed = distance * fps
        else:
            speed = 0

        prev_centers[obj_id] = (center_x, center_y)

        # Determinar color del punto central según velocidad y carril
        point_color = (0, 255, 0)  # Verde por defecto
        if carril == "Izquierdo" and speed < 90:
            point_color = (0, 0, 255)  # Rojo si cumple la condición

        # Dibujar bounding box y mostrar carril
        if show_bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f'Carril: {carril}', (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar punto central con color específico
        point_size = max(5, min((y2 - y1) // 2, y2 - center_y))
        if show_points:
            cv2.circle(frame, (center_x, center_y), point_size, point_color, -1)

    # Mostrar frame procesado
    cv2.imshow('Face Detection & Tracking', frame)
    out.write(frame)

    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
