import cv2
import pytesseract

class PlateRecognition:
    def __init__(self, vehicle_model, plate_model):
        # Modelos para detectar vehículos y placas
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
    
    def detect_vehicles(self, frame):
        """
        Detecta vehículos en un frame.
        Args:
            frame (numpy.ndarray): Frame de video.
        Returns:
            list: Coordenadas de los cuadros delimitadores de vehículos detectados.
        """
        results = self.vehicle_model(frame)
        vehicles = []
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                label = self.vehicle_model.names[int(box.cls)]
                conf = box.conf.item()
                if label == 'car' or label == 'bus':
                    vehicles.append((xmin, ymin, xmax, ymax, label, conf))
        return vehicles

    def extract_plate(self, frame, vehicle_coords):
        """
        Extrae la región de la placa de un vehículo.
        Args:
            frame (numpy.ndarray): Frame de video.
            vehicle_coords (tuple): Coordenadas del vehículo (xmin, ymin, xmax, ymax).
        Returns:
            numpy.ndarray: Imagen de la placa extraída o None si no se detecta ninguna placa.
        """
        xmin, ymin, xmax, ymax = map(int, vehicle_coords)
        vehicle_roi = frame[ymin:ymax, xmin:xmax]
        results = self.plate_model(vehicle_roi)
        for result in results:
            for box in result.boxes:
                pxmin, pymin, pxmax, pymax = map(int, box.xyxy[0])
                plate_image = vehicle_roi[pymin:pymax, pxmin:pxmax]
                return plate_image
        return None

    def recognize_plate_text(self, plate_image):
        """
        Reconoce el texto de una placa utilizando OCR.
        Args:
            plate_image (numpy.ndarray): Imagen de la placa.
        Returns:
            str: Texto reconocido de la placa.
        """
        if plate_image is None or plate_image.size == 0:
            return ""
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        try:
            text = pytesseract.image_to_string(gray, config='--psm 8')
        except pytesseract.TesseractNotFoundError:
            print("Tesseract no está instalado o no está en tu PATH.")
            return ""
        return text.strip()