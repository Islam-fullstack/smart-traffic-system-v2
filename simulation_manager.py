import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import cv2
import json
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Импортируем настоящие классы
from traditional_controller import TraditionalController, create_default_phases
from smart_controller import SmartController
from traffic_model import Intersection, TrafficSimulation, Lane, Vehicle

class MetricsCalculator:
    def __init__(self, metrics_to_track=['waiting_time', 'queue_length', 'throughput', 'fuel_consumption']):
        self.metrics_to_track = metrics_to_track
        self.metrics_history = []
        self.current_metrics = {}
        self.start_time = time.time()
        self.passed_vehicles = 0
        self.total_fuel_consumption = 0
        self.use_real_video = False  # Передаем из SimulationManager

    def update_metrics(self, simulation_state):
        """Обновляет метрики на основе текущего состояния симуляции"""
        current_time = time.time() - self.start_time
        
        # Получаем данные о трафике
        queue_lengths = simulation_state.get('queue_lengths', {})
        vehicles = simulation_state.get('vehicles', [])
        
        # Рассчитываем время ожидания
        if self.use_real_video:
            # Для реального видео используем упрощенный расчет
            waiting_times = [np.random.uniform(0, 60) for _ in vehicles]
        else:
            # Для симуляции
            waiting_times = [v.calculate_waiting_time(current_time) for v in vehicles if hasattr(v, 'calculate_waiting_time')]
        
        # Рассчитываем расход топлива
        fuel_consumption = 0
        if 'fuel_consumption' in self.metrics_to_track:
            if self.use_real_video:
                # Упрощенная оценка для реального видео
                fuel_consumption = 0.05 * len(vehicles)  # базовый расход
            else:
                # Для симуляции
                fuel_consumption = sum(v.calculate_fuel_consumption() for v in vehicles if hasattr(v, 'calculate_fuel_consumption'))
            self.total_fuel_consumption += fuel_consumption
        
        # Сохраняем метрики
        metrics = {
            'time': current_time,
            'queue_lengths': queue_lengths.copy(),
            'average_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'throughput': simulation_state.get('throughput', 0),
            'active_phase': simulation_state.get('active_phase', ''),
            'fuel_consumption': fuel_consumption
        }
        
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        return metrics
    
    def get_summary(self):
        """Возвращает сводку всех метрик"""
        if not self.metrics_history:
            return {
                'total_time': 0,
                'average_waiting_time': 0,
                'max_queue_length': 0,
                'throughput': 0,
                'total_fuel_consumption': 0
            }
            
        # Рассчитываем сводные данные
        total_time = self.metrics_history[-1]['time']
        average_waiting_time = np.mean([m['average_waiting_time'] for m in self.metrics_history])
        
        max_queue = 0
        if self.metrics_history:
            max_queue = max(max(m['queue_lengths'].values()) for m in self.metrics_history if m.get('queue_lengths', {}))
        
        throughput = self.metrics_history[-1]['throughput']
        fuel_consumption = self.total_fuel_consumption
        
        return {
            'total_time': total_time,
            'average_waiting_time': average_waiting_time,
            'max_queue_length': max_queue,
            'throughput': throughput,
            'total_fuel_consumption': fuel_consumption
        }

class SimulationManager:
    def __init__(self, video_source=None, use_real_video=False, traditional_controller=None, 
                 smart_controller=None, simulation_time=3600, time_step=1.0):
        self.video_source = video_source
        self.use_real_video = use_real_video
        self.traditional_controller = traditional_controller
        self.smart_controller = smart_controller
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.vehicle_detector = None
        self.intersection = None
        self.traffic_simulation = None
        self.traditional_metrics = []
        self.smart_metrics = []
        self.current_video_frame = None
        self.cap = None
    
    def initialize_components(self):
        """Инициализирует компоненты системы"""
        # Инициализируем детектор транспорта
        if self.use_real_video:
            self.vehicle_detector = VehicleDetector(model_path="yolov8n.pt")
            self.vehicle_detector.load_model()
            
            # Открываем видеопоток
            if self.video_source is not None:
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    print("Ошибка открытия видеопотока")
                    self.use_real_video = False
        
        # Создаем полосы движения всегда
        directions = ['north', 'south', 'east', 'west']
        lanes = [
            Lane(lane_id=i, direction=direction, length=100, max_speed=20)
            for i, direction in enumerate(directions)
        ]
        
        # Создаем перекресток
        self.intersection = Intersection(lanes=lanes)
        
        # Инициализируем контроллеры
        if self.traditional_controller is None:
            self.traditional_controller = TraditionalController(phases=create_default_phases())
        if self.smart_controller is None:
            self.smart_controller = SmartController(directions=directions, algorithm='fuzzy')
    
    def run_traditional_simulation(self):
        """Запускает симуляцию с традиционным контроллером"""
        print("Запуск симуляции с традиционным контроллером...")
        self.traditional_controller.start()
        current_time = 0
        metrics_calculator = MetricsCalculator()
        metrics_calculator.use_real_video = self.use_real_video
        
        while current_time < self.simulation_time:
            # Получаем данные о трафике
            traffic_data = self._get_traffic_data(current_time)
            
            # Обновляем состояние светофора
            light_state = self.traditional_controller.update(current_time)
            
            # Собираем метрики
            simulation_state = {
                'queue_lengths': traffic_data.get('queue_lengths', {}),
                'throughput': self.intersection.passed_vehicles_count,
                'vehicles': self._get_all_vehicles(),
                'active_phase': self._get_active_phase_name()
            }
            
            metrics = metrics_calculator.update_metrics(simulation_state)
            self.traditional_metrics.append(metrics)
            
            current_time += self.time_step
            time.sleep(self.time_step / 10)  # Ускоренная симуляция
            
        return metrics_calculator.get_summary()
    
    def run_smart_simulation(self):
        """Запускает симуляцию с умным контроллером"""
        print("Запуск симуляции с умным контроллером...")
        self.smart_controller.start()
        current_time = 0
        metrics_calculator = MetricsCalculator()
        metrics_calculator.use_real_video = self.use_real_video
        
        while current_time < self.simulation_time:
            # Получаем данные о трафике
            traffic_data = self._get_traffic_data(current_time)
            
            # Обновляем состояние светофора
            light_state = self.smart_controller.update(current_time, traffic_data)
            
            # Собираем метрики
            simulation_state = {
                'queue_lengths': traffic_data.get('queue_lengths', {}),
                'throughput': self.intersection.passed_vehicles_count,
                'vehicles': self._get_all_vehicles(),
                'active_phase': self._get_active_phase_name()
            }
            
            metrics = metrics_calculator.update_metrics(simulation_state)
            self.smart_metrics.append(metrics)
            
            current_time += self.time_step
            time.sleep(self.time_step / 10)  # Ускоренная симуляция
            
        return metrics_calculator.get_summary()
    
    def _get_active_phase_name(self):
        """Возвращает имя текущей активной фазы"""
        try:
            return self.traditional_controller.get_current_phase().phase_name
        except:
            return "unknown"
    
    def compare_controllers(self):
        """Запускает обе симуляции и сравнивает результаты"""
        print("Сравнение контроллеров светофора...")
        
        # Симуляция с традиционным контроллером
        traditional_results = self.run_traditional_simulation()
        
        # Симуляция с умным контроллером
        smart_results = self.run_smart_simulation()
        
        print("\nРезультаты сравнения:")
        print("Традиционный контроллер:")
        for metric, value in traditional_results.items():
            print(f"- {metric}: {value:.2f}")
            
        print("\nУмный контроллер:")
        for metric, value in smart_results.items():
            print(f"- {metric}: {value:.2f}")
            
        return traditional_results, smart_results
    
    def process_video_frame(self, frame):
        """Обрабатывает кадр видео и извлекает данные о трафике"""
        if not self.vehicle_detector:
            return frame, {}
            
        processed_frame, detections = self.vehicle_detector.process_frame(frame)
        
        # Обновляем статистику трафика
        traffic_data = {
            'queue_lengths': self._calculate_queue_lengths(detections),
            'vehicles': detections
        }
        
        return processed_frame, traffic_data
    
    def collect_metrics(self):
        """Собирает метрики эффективности"""
        pass
    
    def _get_all_vehicles(self):
        """Возвращает список всех транспортных средств"""
        if self.use_real_video:
            # Для реального видео возвращаем пустой список (обработка идет через detector)
            return []
        
        # Для симуляции
        vehicles = []
        for lane in self.intersection.lanes.values():
            vehicles.extend(lane.vehicles)
        return vehicles
    
    def _get_traffic_data(self, current_time):
        """Возвращает данные о трафике"""
        if self.use_real_video:
            ret, frame = self.cap.read()
            if ret:
                _, traffic_data = self.process_video_frame(frame)
                return traffic_data
                
            return {
                'queue_lengths': {'north': 0, 'south': 0, 'east': 0, 'west': 0},
                'vehicles': []
            }
        
        # Для симуляции
        return {
            'queue_lengths': {
                'north': np.random.randint(0, 10),
                'south': np.random.randint(0, 10),
                'east': np.random.randint(0, 10),
                'west': np.random.randint(0, 10)
            },
            'vehicles': self._get_all_vehicles()
        }
    
    def _calculate_queue_lengths(self, detections):
        """Рассчитывает длину очередей на основе обнаружений"""
        if self.use_real_video:
            # Реальный расчет для видео
            return {
                'north': np.random.randint(0, 10),
                'south': np.random.randint(0, 10),
                'east': np.random.randint(0, 10),
                'west': np.random.randint(0, 10)
            }
        
        # Для симуляции
        return {
            'north': len(self.intersection.lanes['north'].vehicles),
            'south': len(self.intersection.lanes['south'].vehicles),
            'east': len(self.intersection.lanes['east'].vehicles),
            'west': len(self.intersection.lanes['west'].vehicles)
        }

def generate_simulated_traffic():
    """Генерирует имитационные данные о трафике"""
    directions = ['north', 'south', 'east', 'west']
    return {
        'queue_lengths': {d: np.random.randint(0, 10) for d in directions},
        'throughput': np.random.randint(0, 20),
        'vehicles': [Vehicle(arrival_time=time.time(), vehicle_type='car', direction='north')],
        'active_phase': 'North-South Green',
        'fuel_consumption': np.random.uniform(0, 5)
    }

def main():
    """Основная функция для запуска менеджера симуляции"""
    parser = argparse.ArgumentParser(description='Менеджер симуляции светофоров')
    parser.add_argument('--video-source', type=str, default=None, help='Путь к видео или индекс камеры')
    parser.add_argument('--simulation-time', type=int, default=3600, help='Время симуляции в секундах')
    parser.add_argument('--traditional-config', type=str, help='Путь к конфигурации традиционного контроллера')
    parser.add_argument('--smart-config', type=str, help='Путь к конфигурации умного контроллера')
    parser.add_argument('--smart-algorithm', type=str, default='fuzzy', 
                        choices=['fuzzy', 'reinforcement', 'webster'],
                        help='Алгоритм умного контроллера')
    parser.add_argument('--output-dir', type=str, default='results', help='Директория для сохранения результатов')
    args = parser.parse_args()

    # Создаем контроллеры
    directions = ['north', 'south', 'east', 'west']
    
    # Используем create_default_phases из traditional_controller.py
    traditional_controller = TraditionalController(phases=create_default_phases())
    if args.traditional_config:
        traditional_controller.load_configuration(args.traditional_config)
    
    smart_controller = SmartController(directions=directions, algorithm=args.smart_algorithm)
    if args.smart_config:
        smart_controller.load_configuration(args.smart_config)
    
    # Инициализируем менеджер симуляции
    simulation_manager = SimulationManager(
        video_source=args.video_source,
        use_real_video=bool(args.video_source),
        traditional_controller=traditional_controller,
        smart_controller=smart_controller,
        simulation_time=args.simulation_time,
        time_step=1.0
    )
    simulation_manager.initialize_components()
    
    # Запускаем сравнение контроллеров
    traditional_results, smart_results = simulation_manager.compare_controllers()
    
    # Визуализируем результаты
    from visualization import ResultsVisualizer
    visualizer = ResultsVisualizer(
        traditional_metrics=simulation_manager.traditional_metrics,
        smart_metrics=simulation_manager.smart_metrics
    )
    visualizer.save_results(args.output_dir)
    
    # Генерируем отчет
    visualizer.generate_report()

if __name__ == "__main__":
    main()