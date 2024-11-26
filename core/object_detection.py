import cv2
from ultralytics import YOLO
import os
class ObjectDetector:
    def __init__(self, model):
        """
        Inicializa el detector de objetos con un modelo YOLOv8 preentrenado.
        Args:
            model (YOLO): Instancia del modelo YOLOv8.
        """
        self.model = model

    def detect_objects(self, image):
        """
        Detecta objetos en una imagen utilizando el modelo YOLOv8.
        Args:
            image (numpy.ndarray): Imagen en la que se detectarán los objetos.
        Returns:
            list: Lista de cuadros delimitadores y etiquetas de los objetos detectados.
        """
        results = self.model(image)
        detections = []
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls)]
                conf = box.conf.item()
                detections.append((xmin, ymin, xmax, ymax, label, conf))
        return detections
    def draw_detections(self, image, detections):
        """
        Dibuja los cuadros delimitadores y etiquetas en la imagen.
        Args:
            image (numpy.ndarray): Imagen original.
            detections (list): Lista de detecciones con cuadros delimitadores y etiquetas.
        Returns:
            numpy.ndarray: Imagen con cuadros delimitadores y etiquetas dibujados.
        """
        for (xmin, ymin, xmax, ymax, label, conf) in detections:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

# Ejemplo de uso
if __name__ == "__main__":
    detector = ObjectDetector("models/detection/yolov8s.pt")
    image_folder = "/home/p0wden/Documents/IA/coco/val2017"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))][:20]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        detections = detector.detect_objects(image)
        image_with_detections = detector.draw_detections(image, detections)
        
        # Mostrar la imagen con detecciones
        cv2.imshow(f"Detections - {image_file}", image_with_detections)
        cv2.waitKey(0)
        cv2.destroyAllWindows()