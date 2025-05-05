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

from traditional_controller import TraditionalController, create_default_phases
from smart_controller import SmartController

# from smart_controller import SmartController, create_default_phases
from traffic_model import Intersection, TrafficSimulation, Lane, Vehicle
from video_processor import VehicleDetector

# Импортируем необходимые классы из других модулей
# Замените на реальные импорты из ваших файлов
class VehicleDetector:
    def __init__(self, *args, **kwargs):
        pass
    
    def load_model(self):
        pass
    
    def process_frame(self, frame):
        return frame, []

# class TraditionalController:
#     def __init__(self, *args, **kwargs):
#         pass
    
#     def start(self):
#         pass
    
#     def update(self, current_time=None):
#         return {'north': 'green', 'south': 'red', 'east': 'red', 'west': 'red'}

class SmartController:
    def __init__(self, directions, *args, **kwargs):
        self.directions = directions
    
    def start(self):
        pass
    
    def update(self, current_time, traffic_data):
        return {'north': 'green', 'south': 'red', 'east': 'red', 'west': 'red'}

class Intersection:
    def __init__(self, *args, **kwargs):
        pass

class TrafficSimulation:
    def __init__(self, intersection, *args, **kwargs):
        pass
    
    def run_simulation(self):
        return []
    
    def get_results(self):
        return {
            'total_passed': 100,
            'average_waiting_time': 30,
            'total_time': 3600
        }

class Lane:
    def __init__(self, *args, **kwargs):
        self.vehicles = []

class Vehicle:
    def __init__(self, arrival_time, *args, **kwargs):
        self.arrival_time = arrival_time
        self.passed = False

class MetricsCalculator:
    def __init__(self, metrics_to_track=['waiting_time', 'queue_length', 'throughput']):
        self.metrics_to_track = metrics_to_track
        self.metrics_history = []
        self.current_metrics = {}
        self.start_time = time.time()
        self.passed_vehicles = 0
        self.total_fuel_consumption = 0
        
    def update_metrics(self, simulation_state):
        """Обновляет метрики на основе текущего состояния симуляции"""
        current_time = time.time() - self.start_time
        
        # Проверяем, что очередь есть
        queue_lengths = simulation_state.get('queue_lengths', {})
        if not queue_lengths:
            queue_lengths = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # Вычисляем метрики
        metrics = {
            'time': current_time,
            'queue_lengths': queue_lengths.copy(),
            'throughput': simulation_state.get('throughput', 0),
            'active_phase': simulation_state.get('active_phase', '')
        }
        
        # Рассчитываем среднее время ожидания
        vehicles = simulation_state.get('vehicles', [])
        waiting_times = [v.calculate_waiting_time(current_time) for v in vehicles if hasattr(v, 'calculate_waiting_time')]
        metrics['average_waiting_time'] = np.mean(waiting_times) if waiting_times else 0
        
        # Рассчитываем расход топлива
        fuel_consumption = sum(v.calculate_fuel_consumption() for v in vehicles if hasattr(v, 'calculate_fuel_consumption'))
        self.total_fuel_consumption += fuel_consumption
        metrics['fuel_consumption'] = fuel_consumption
        
        # Сохраняем метрики
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        
        return metrics
    
    def calculate_average_waiting_time(self, vehicles):
        """Вычисляет среднее время ожидания"""
        current_time = time.time() - self.start_time
        if self.use_real_video:
            # Для реального видео используем упрощенный расчет
            return np.mean([np.random.uniform(0, 60) for _ in vehicles]) if vehicles else 0
        
        # Для симуляции
        waiting_times = [v.calculate_waiting_time(current_time) for v in vehicles if hasattr(v, 'calculate_waiting_time')]
        return np.mean(waiting_times) if waiting_times else 0
    
    def calculate_throughput(self, passed_vehicles, time_interval=60):
        """Вычисляет пропускную способность"""
        return passed_vehicles / (time_interval / 60)  # ТС/мин
    
    def calculate_fuel_consumption(self, vehicles):
        """Оценивает общий расход топлива"""
        # Базовый расход для холостого хода и остановок
        base_consumption = 0.05 * len(vehicles)  # л/мин на машину
        acceleration_penalty = 0.02 * sum(1 for v in vehicles if hasattr(v, 'is_accelerating') and v.is_accelerating)
        
        return base_consumption + acceleration_penalty
    
    def get_summary(self):
        """Возвращает сводку всех метрик"""
        summary = {
            'total_time': self.metrics_history[-1]['time'] if self.metrics_history else 0,
            'total_vehicles': len([m for m in self.metrics_history if m.get('queue_lengths', {})]),
            'average_waiting_time': np.mean([m['average_waiting_time'] for m in self.metrics_history]),
            'max_queue_length': max(
                [max(q.values()) for q in [m['queue_lengths'] for m in self.metrics_history]] or [0]
            ),
            'throughput': self.metrics_history[-1]['throughput'] if self.metrics_history else 0,
            'total_fuel_consumption': self.total_fuel_consumption
        }
        return summary

class ResultsVisualizer:
    def __init__(self, traditional_metrics, smart_metrics):
        self.traditional_metrics = traditional_metrics
        self.smart_metrics = smart_metrics
        self.output_dir = None
    
    def plot_waiting_time_comparison(self):
        """Создает график сравнения времени ожидания"""
        plt.figure(figsize=(12, 6))
        
        # Для традиционного контроллера
        times_traditional = [m['time'] for m in self.traditional_metrics]
        wait_times_traditional = [m['average_waiting_time'] for m in self.traditional_metrics]
        plt.plot(times_traditional, wait_times_traditional, label='Традиционный контроллер', marker='o', linestyle='-')
        
        # Для умного контроллера
        times_smart = [m['time'] for m in self.smart_metrics]
        wait_times_smart = [m['average_waiting_time'] for m in self.smart_metrics]
        plt.plot(times_smart, wait_times_smart, label='Умный контроллер', marker='s', linestyle='--')
        
        plt.xlabel('Время (сек)')
        plt.ylabel('Среднее время ожидания (сек)')
        plt.title('Сравнение времени ожидания')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'waiting_time_comparison.png'))
        plt.show()
    
    def plot_queue_length_comparison(self):
        """Создает график сравнения длин очередей"""
        plt.figure(figsize=(12, 6))
        
        times_traditional = [m['time'] for m in self.traditional_metrics]
        queue_lengths_traditional = [max(m['queue_lengths'].values()) for m in self.traditional_metrics]
        plt.plot(times_traditional, queue_lengths_traditional, label='Традиционный контроллер', marker='o', linestyle='-')
        
        times_smart = [m['time'] for m in self.smart_metrics]
        queue_lengths_smart = [max(m['queue_lengths'].values()) for m in self.smart_metrics]
        plt.plot(times_smart, queue_lengths_smart, label='Умный контроллер', marker='s', linestyle='--')
        
        plt.xlabel('Время (сек)')
        plt.ylabel('Максимальная длина очереди')
        plt.title('Сравнение длины очередей')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'queue_length_comparison.png'))
        plt.show()
    
    def plot_throughput_comparison(self):
        """Создает график сравнения пропускной способности"""
        plt.figure(figsize=(12, 6))
        
        times_traditional = [m['time'] for m in self.traditional_metrics]
        throughput_traditional = [m['throughput'] for m in self.traditional_metrics]
        plt.plot(times_traditional, throughput_traditional, label='Традиционный контроллер', marker='o', linestyle='-')
        
        times_smart = [m['time'] for m in self.smart_metrics]
        throughput_smart = [m['throughput'] for m in self.smart_metrics]
        plt.plot(times_smart, throughput_smart, label='Умный контроллер', marker='s', linestyle='--')
        
        plt.xlabel('Время (сек)')
        plt.ylabel('Пропускная способность')
        plt.title('Сравнение пропускной способности')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if self.output_dir:
            plt.savefig(os.path.join(self.output_dir, 'throughput_comparison.png'))
        plt.show()
    
    def generate_report(self):
        """Создает полный отчет с графиками и таблицами"""
        print("\n=== Сравнительный анализ ===")
        
        # Сводка метрик
        trad_summary = {
            'total_time': self.traditional_metrics[-1]['time'] if self.traditional_metrics else 0,
            'average_waiting_time': np.mean([m['average_waiting_time'] for m in self.traditional_metrics]) 
                                   if self.traditional_metrics else 0,
            'max_queue': max([max(m['queue_lengths'].values()) for m in self.traditional_metrics]) 
                        if self.traditional_metrics else 0,
            'throughput': self.traditional_metrics[-1]['throughput'] if self.traditional_metrics else 0
        }
        
        smart_summary = {
            'total_time': self.smart_metrics[-1]['time'] if self.smart_metrics else 0,
            'average_waiting_time': np.mean([m['average_waiting_time'] for m in self.smart_metrics]) 
                                   if self.smart_metrics else 0,
            'max_queue': max([max(m['queue_lengths'].values()) for m in self.smart_metrics]) 
                        if self.smart_metrics else 0,
            'throughput': self.smart_metrics[-1]['throughput'] if self.smart_metrics else 0
        }
        
        print("\nТрадиционный контроллер:")
        for metric, value in trad_summary.items():
            print(f"- {metric}: {value:.2f}")
        
        print("\nУмный контроллер:")
        for metric, value in smart_summary.items():
            print(f"- {metric}: {value:.2f}")
        
        # Визуализируем графики
        self.plot_waiting_time_comparison()
        self.plot_queue_length_comparison()
        self.plot_throughput_comparison()
    
    def save_results(self, output_dir):
        """Сохраняет результаты в указанную директорию"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем метрики в JSON
        results = {
            'traditional_controller': self.traditional_metrics,
            'smart_controller': self.smart_metrics,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'traditional': self._generate_summary(self.traditional_metrics),
                'smart': self._generate_summary(self.smart_metrics)
            }
        }
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        # Сохраняем графики
        self.output_dir = output_dir
        self.plot_waiting_time_comparison()
        self.plot_queue_length_comparison()
        self.plot_throughput_comparison()
    
    def _generate_summary(self, metrics):
        """Генерирует сводные данные из метрик"""
        if not metrics:
            return {}
            
        total_time = metrics[-1]['time']
        average_waiting_time = np.mean([m['average_waiting_time'] for m in metrics])
        max_queue = max([max(m['queue_lengths'].values()) for m in metrics])
        throughput = metrics[-1]['throughput']
        
        return {
            'total_time': total_time,
            'average_waiting_time': average_waiting_time,
            'max_queue_length': max_queue,
            'throughput': throughput,
            'fuel_consumption': sum(m.get('fuel_consumption', 0) for m in metrics)
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
        lanes = [
            Lane(lane_id=i, direction=direction, length=100, max_speed=20)
            for i, direction in enumerate(['north', 'south', 'east', 'west'])
        ]

        # Создаем перекресток
        self.intersection = type('obj', (object,), {
            'passed_vehicles_count': 0,
            'lanes': {d: type('obj', (object,), {'vehicles': []}) for d in ['north', 'south', 'east', 'west']}
        })
        
        # Создаем контроллеры
        if self.traditional_controller is None:
            self.traditional_controller = TraditionalController(phases=create_default_phases())
        if self.smart_controller is None:
            self.smart_controller = SmartController(directions=['north', 'south', 'east', 'west'])
    
    def run_traditional_simulation(self):
        """Запускает симуляцию с традиционным контроллером"""
        print("Запуск симуляции с традиционным контроллером...")
        self.traditional_controller.start()
        current_time = 0
        metrics_calculator = MetricsCalculator()
        
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
                'active_phase': self.traditional_controller.get_current_phase().phase_name
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
        
        while current_time < self.simulation_time:
            # Получаем данные о трафике
            traffic_data = self._get_traffic_data(current_time)
            
            # Обновляем состояние светофора
            light_state = self.smart_controller.update(current_time, traffic_data)
            
            # Собираем метрики
            simulation_state = {
                'queue_lengths': traffic_data.get('queue_lengths', {}),
                'throughput': self.intersection.passed_vehicles_count,
                'vehicles': self._get_all_vehicles()
            }
            
            metrics = metrics_calculator.update_metrics(simulation_state)
            self.smart_metrics.append(metrics)
            
            current_time += self.time_step
            time.sleep(self.time_step / 10)  # Ускоренная симуляция
            
        return metrics_calculator.get_summary()
    
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
            print(f"  {metric}: {value:.2f}")
            
        print("\nУмный контроллер:")
        for metric, value in smart_results.items():
            print(f"  {metric}: {value:.2f}")
            
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
            # Для реального видео используем данные с детектора
            return []
        
        # Для симуляции
        vehicles = []
        for lane in self.intersection.lanes.values():
            vehicles.extend(lane.vehicles)
        return vehicles
    
    def _get_traffic_data(self, current_time):
        """Возвращает данные о трафике"""
        if self.use_real_video:
            # Получаем данные с видео
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
            # Реальный расчет для видео (пример)
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

# def create_default_phases():
#     """Создает стандартные фазы для традиционного контроллера."""
#     return [
#         {
#             "phase_id": 0,
#             "phase_name": "North-South Green",
#             "durations": {"green": 30, "yellow": 5, "red": 0},
#             "directions": {
#                 "north": "green",
#                 "south": "green",
#                 "east": "red",
#                 "west": "red"
#             }
#         },
#         {
#             "phase_id": 1,
#             "phase_name": "East-West Green",
#             "durations": {"green": 30, "yellow": 5, "red": 0},
#             "directions": {
#                 "north": "red",
#                 "south": "red",
#                 "east": "green",
#                 "west": "green"
#             }
#         }
#     ]

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
    
    # Создаем полосы движения
    lanes = [
        Lane(lane_id=i, direction=direction, length=100, max_speed=20)
        for i, direction in enumerate(directions)
    ]
    
    # Создаем перекресток
    intersection = Intersection(lanes=lanes, traffic_light=None)
    
    traditional_controller = TraditionalController(phases=create_default_phases())
    if args.traditional_config:
        traditional_controller.load_configuration(args.traditional_config)
    
    smart_controller = SmartController(
        directions=directions, 
        algorithm=args.smart_algorithm
    )
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
    
    # Инициализируем компоненты
    simulation_manager.initialize_components()
    
    # Запускаем сравнение контроллеров
    traditional_results, smart_results = simulation_manager.compare_controllers()
    
    # Визуализируем результаты
    visualizer = ResultsVisualizer(
        traditional_metrics=simulation_manager.traditional_metrics,
        smart_metrics=simulation_manager.smart_metrics
    )
    visualizer.save_results(args.output_dir)
    
    # Генерируем отчет
    visualizer.generate_report()

if __name__ == "__main__":
    main()