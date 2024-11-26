import threading
from ultralytics import YOLO

def process_vehicle(frame, xmin, ymin, xmax, ymax):
    plate_model = YOLO("yolov8n.pt")  # Instancia el modelo dentro del hilo
    vehicle_roi = frame[ymin:ymax, xmin:xmax]
    results = plate_model(vehicle_roi)
    print("Resultados:", results)

# Ejecuta el hilo
frame = ...  # Coloca tu frame aquí
thread = threading.Thread(target=process_vehicle, args=(frame, 0, 0, 100, 100))
thread.start()
thread.join()
