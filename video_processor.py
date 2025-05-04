import cv2
import numpy as np
from ultralytics import YOLO
import argparse

class VehicleDetector:
    def __init__(self, model_path, confidence_threshold=0.5, classes=['car', 'truck', 'bus', 'motorcycle']):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.model = None
        self.class_ids = []

    def load_model(self):
        """Загружает модель YOLO и определяет соответствующие классы"""
        self.model = YOLO(self.model_path)
        names = self.model.names
        self.class_ids = [k for k, v in names.items() if v in self.classes]

    def process_frame(self, frame):
        """Обрабатывает кадр и возвращает аннотированный кадр и список обнаруженных объектов"""
        results = self.model(frame, conf=self.confidence_threshold, classes=self.class_ids)
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                xyxy = box.xyxy[0].tolist()
                class_name = self.model.names[cls]
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': xyxy
                })
                # Рисуем рамку и подпись
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return annotated_frame, detections

    def count_vehicles_by_direction(self, detections, regions):
        """Подсчитывает транспортные средства в каждом регионе"""
        counts = {direction: 0 for direction in regions}
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            for direction, (rx1, ry1, rx2, ry2) in regions.items():
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    counts[direction] += 1
        return counts

class VideoProcessor:
    def __init__(self, source, detector, roi_regions, output_path=None):
        self.source = source
        self.detector = detector
        self.roi_regions = roi_regions
        self.output_path = output_path
        self.cap = None
        self.writer = None
        self.traffic_data = {}

    def start_processing(self):
        """Запускает обработку видеопотока"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("Ошибка открытия видеопотока")
            return
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame, detections = self.detector.process_frame(frame)
                counts = self.detector.count_vehicles_by_direction(detections, self.roi_regions)
                self.traffic_data = counts
                
                # Отображение счетчиков на кадре
                for i, (direction, count) in enumerate(counts.items()):
                    text = f"{direction}: {count}"
                    cv2.putText(processed_frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                if self.output_path:
                    self.writer.write(processed_frame)
                
                cv2.imshow('Processed Video', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release_resources()
        
        return self.traffic_data

    def release_resources(self):
        """Освобождает ресурсы"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def calibrate_roi(self):
        """Интерактивная калибровка регионов интереса"""
        def draw_rectangle(event, x, y, flags, param):
            nonlocal drawing, ix, iy, frame_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp = frame_copy.copy()
                    cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
                    cv2.imshow('Calibrate ROI', temp)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                regions[direction] = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
                cv2.rectangle(frame_copy, (min(ix, x), min(iy, y)), 
                             (max(ix, x), max(iy, y)), (0, 255, 0), 2)
                print(f"Region {direction} set to {regions[direction]}")

        # Захват кадра
        self.cap = cv2.VideoCapture(self.source)
        ret, frame = self.cap.read()
        if not ret:
            print("Не удалось получить кадр для калибровки ROI")
            self.cap.release()
            return
        
        frame_copy = frame.copy()
        regions = {}
        directions = ['north', 'south', 'east', 'west']
        drawing = False
        ix, iy = -1, -1
        
        try:
            for direction in directions:
                drawing = False
                ix, iy = -1, -1
                cv2.namedWindow('Calibrate ROI')
                cv2.setMouseCallback('Calibrate ROI', draw_rectangle)
                print(f"Выберите регион для {direction}. Нажмите Enter после выделения.")
                
                while True:
                    cv2.imshow('Calibrate ROI', frame_copy)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 13:  # Enter
                        break
        finally:
            self.roi_regions = regions
            self.release_resources()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Путь к видео или индекс камеры')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Путь к модели YOLO')
    parser.add_argument('--confidence', type=float, default=0.5, help='Порог уверенности')
    parser.add_argument('--output', type=str, default=None, help='Путь для сохранения видео')
    args = parser.parse_args()

    # Инициализация детектора
    detector = VehicleDetector(model_path=args.model, confidence_threshold=args.confidence)
    detector.load_model()

    # Создание регионов интереса
    roi_regions = {}  # Может быть, изначально пустой, затем калибровать
    
    # Инициализация процессора видео
    source = int(args.source) if args.source.isdigit() else args.source
    processor = VideoProcessor(
        source=source,
        detector=detector,
        roi_regions=roi_regions,
        output_path=args.output
    )
    
    # Калибровка ROI
    print("Калибровка регионов интереса...")
    processor.calibrate_roi()
    
    # Запуск обработки
    print("Запуск обработки видео...")
    traffic_data = processor.start_processing()
    
    print("Данные о трафике:", traffic_data)

if __name__ == "__main__":
    main()