from core.video_processing import load_video
from core.object_detection import ObjectDetector
from core.plate_recognition import PlateRecognition
from ultralytics import YOLO
import os
import threading

def process_vehicle(frame, vehicle, plate_recognizer):
    xmin, ymin, xmax, ymax, label, conf = vehicle
    plate_image = plate_recognizer.extract_plate(frame, (xmin, ymin, xmax, ymax))
    if plate_image is not None:
        plate_text = plate_recognizer.recognize_plate_text(plate_image)
        if plate_text:
            print(f"Placa detectada: {plate_text}")
        else:
            print("No se pudo reconocer el texto de la placa.")
    else:
        print("No se detectó ninguna placa en el vehículo.")

def process_person(frame, person):
    xmin, ymin, xmax, ymax, label, conf = person
    # Aquí puedes agregar el procesamiento específico para personas
    print(f"Persona detectada en: ({xmin}, {ymin}, {xmax}, {ymax})")

def main(video_path):
    if not os.path.isfile(video_path):
        print(f"El archivo {video_path} no existe.")
        return
    
    vehicle_model = YOLO("models/detection/yolov8s.pt")
    plate_model = YOLO("models/detection/license_plate_detector.pt")
    vehicle_detector = ObjectDetector(vehicle_model)
    plate_recognizer = PlateRecognition(vehicle_model, plate_model)

    print(f"Procesando video: {video_path}")
    for frame in load_video(video_path):
        if frame is None:
            continue
        
        detections = vehicle_detector.detect_objects(frame)
        vehicles = [d for d in detections if d[4] == 'car' or d[4] == 'bus']
        persons = [d for d in detections if d[4] == 'person']

        # Procesar vehículos en hilos separados
        for vehicle in vehicles:
            vehicle_thread = threading.Thread(target=process_vehicle, args=(frame, vehicle, plate_recognizer))
            vehicle_thread.start()

        # Procesar personas en hilos separados
        for person in persons:
            person_thread = threading.Thread(target=process_person, args=(frame, person))
            person_thread.start()

if __name__ == "__main__":
    VIDEO_PATH_Carro = "/home/p0wden/Documents/IA/Vigilancia/datasets/traffic_video_modified.mp4"
    main(VIDEO_PATH_Carro)