import cv2

def load_video(file_path):
    """
    Carga un video desde un archivo local.
    Args:
        file_path (str): Ruta al archivo de video.
    Yields:
        frame (numpy.ndarray): Frame procesado del video.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"El archivo {file_path} no se pudo abrir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        yield preprocess_frame(frame)
    cap.release()

def preprocess_frame(frame):
    """
    Aplica preprocesamiento básico a un frame.
    Args:
        frame (numpy.ndarray): Frame original.
    Returns:
        numpy.ndarray: Frame preprocesado.
    """
    # Asegurarse de que la imagen tiene 3 canales (color)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    # Redimensionar el frame a un tamaño estándar (opcional)
    frame = cv2.resize(frame, (640, 640))
    
    return frame