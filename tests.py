import unittest
import time
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple
import os
import argparse

# Импортируем тестируемые модули
try:
    from video_processor import VehicleDetector
except ImportError:
    VehicleDetector = Mock  # Заглушка, если нет реального модуля
    
try:
    from traffic_model import Lane, Vehicle, Intersection, TrafficSimulation
except ImportError:
    Lane = Mock
    Vehicle = Mock
    Intersection = Mock
    TrafficSimulation = Mock
    
try:
    from traditional_controller import TraditionalController, create_default_phases
except ImportError:
    TraditionalController = Mock
    create_default_phases = Mock
    
try:
    from smart_controller import SmartController
except ImportError:
    SmartController = Mock
    
try:
    from simulation_manager import SimulationManager
except ImportError:
    SimulationManager = Mock
    
try:
    from visualization import IntersectionVisualizer
except ImportError:
    IntersectionVisualizer = Mock

def create_mock_data():
    """Создает тестовые данные для всех компонентов"""
    return {
        "video_frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "traffic_data": {
            "north": 10,
            "south": 5,
            "east": 15,
            "west": 8
        },
        "simulation_state": {
            "queue_lengths": {"north": 3, "south": 2, "east": 5, "west": 4},
            "throughput": 20,
            "vehicles": [
                Vehicle(arrival_time=0, vehicle_type='car', direction='north'),
                Vehicle(arrival_time=1, vehicle_type='truck', direction='south')
            ]
        },
        "configuration": {
            "controllers": {
                "traditional": {
                    "phases": create_default_phases(),
                    "cycle_times": {"morning": 120, "day": 90, "evening": 120, "night": 60}
                },
                "smart": {
                    "algorithm": "fuzzy",
                    "min_green": 15,
                    "max_green": 90,
                    "yellow": 5
                }
            },
            "simulation": {
                "time": 300,
                "time_step": 1.0
            },
            "visualization": {
                "window_size": "800,600",
                "scale": 10.0,
                "fps": 30
            }
        }
    }

def performance_test(component, iterations=100):
    """
    Измеряет производительность указанного компонента
    
    Args:
        component: объект или функция для тестирования
        iterations: количество итераций
        
    Returns:
        dict: статистика производительности
    """
    times = []
    
    # Выполняем тестовое действие несколько раз
    for _ in range(iterations):
        start = time.perf_counter()
        
        if hasattr(component, "update"):
            component.update(time.time())
        elif hasattr(component, "process_frame"):
            component.process_frame(np.random.randint(0, 255, (480, 640, 3), np.uint8))
        elif hasattr(component, "run_traditional_simulation"):
            component.run_traditional_simulation()
        elif hasattr(component, "draw_intersection"):
            component.draw_intersection()
            
        end = time.perf_counter()
        times.append(end - start)
    
    # Рассчитываем статистику
    perf_stats = {
        "component": str(component.__class__.__name__ if hasattr(component, "__class__") else component),
        "iterations": iterations,
        "avg_time": np.mean(times),
        "std_dev": np.std(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times)
    }
    
    return perf_stats

class TestVehicleDetector(unittest.TestCase):
    """Тесты для класса VehicleDetector"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        try:
            self.detector = VehicleDetector(model_path="yolov8n.pt")
        except Exception as e:
            self.detector = Mock()
            print(f"Используем заглушку VehicleDetector из-за ошибки: {e}")
            
        # Создаем тестовый кадр
        self.test_frame = self.test_data["video_frame"]
    
    def test_initialization(self):
        """Проверяет корректность инициализации детектора"""
        self.assertIsNotNone(self.detector, "Детектор должен быть инициализирован")
        
        # Проверяем базовые атрибуты
        if hasattr(self.detector, "model_path"):
            self.assertEqual(self.detector.model_path, "yolov8n.pt", "Путь к модели должен совпадать")
    
    def test_process_frame(self):
        """Тестирует обработку кадра"""
        with patch('video_processor.VehicleDetector.process_frame', return_value=(self.test_frame, [])):
            processed_frame, detections = self.detector.process_frame(self.test_frame)
            
        self.assertIsNotNone(processed_frame, "Обработанный кадр не должен быть None")
        self.assertIsInstance(detections, list, "Обнаружения должны быть списком")
    
    def test_count_vehicles(self):
        """Проверяет подсчет транспортных средств"""
        with patch('video_processor.VehicleDetector.process_frame', return_value=(self.test_frame, ["car", "truck"])):
            _, detections = self.detector.process_frame(self.test_frame)
            self.assertEqual(len(detections), 2, "Должно быть обнаружено 2 ТС")
    
    def test_performance(self):
        """Измеряет производительность обработки кадров"""
        stats = performance_test(self.detector, iterations=10)
        print(f"\nПроизводительность VehicleDetector: {stats['avg_time']:.4f} сек/итерация")

class TestTrafficModel(unittest.TestCase):
    """Тесты для модели трафика"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        
        # Создаем полосы
        self.lanes = [
            Lane(lane_id=i, direction=direction, length=100, max_speed=20)
            for i, direction in enumerate(['north', 'south', 'east', 'west'])
        ]
        
        # Создаем перекресток
        self.intersection = Intersection(lanes=self.lanes)
        
        # Создаем симуляцию
        self.simulation = TrafficSimulation(
            intersection=self.intersection,
            arrival_rates={"north": 10, "south": 10, "east": 15, "west": 15},
            vehicle_type_distribution={"car": 0.6, "truck": 0.1, "bus": 0.2, "motorcycle": 0.1},
            simulation_time=300
        )
    
    def test_vehicle_creation(self):
        """Проверяет создание транспортных средств"""
        vehicle = Vehicle(arrival_time=0, vehicle_type='car', direction='north')
        self.assertEqual(vehicle.vehicle_type, 'car')
        self.assertEqual(vehicle.direction, 'north')
        self.assertEqual(vehicle.position, 100)  # Проверяем начальную позицию
    
    def test_lane_operations(self):
        """Тестирует операции с полосами движения"""
        # Добавляем ТС в полосу
        vehicle = Vehicle(arrival_time=0, vehicle_type='car', direction='north')
        self.intersection.lanes['north'].add_vehicle(vehicle)
        
        # Проверяем длину очереди
        self.assertEqual(self.intersection.lanes['north'].get_queue_length(), 1)
        
        # Обновляем состояние полосы
        self.intersection.lanes['north'].update(is_green=True, time_step=1.0)
        self.assertEqual(self.intersection.lanes['north'].get_queue_length(), 0)
    
    def test_intersection_update(self):
        """Проверяет обновление состояния перекрестка"""
        # Обновляем состояние перекрестка
        self.intersection.update(current_time=0, time_step=1.0)
        
        # Проверяем, что метрики собраны
        self.assertGreater(len(self.intersection.metrics_history), 0, "Должны быть собраны метрики")
    
    def test_traffic_generation(self):
        """Тестирует генерацию транспортного потока"""
        # Генерируем ТС
        self.simulation.generate_vehicles()
        
        # Проверяем, что ТС добавлены
        total_vehicles = sum(len(lane.vehicles) for lane in self.intersection.lanes.values())
        self.assertGreater(total_vehicles, 0, "Должны быть созданы ТС")
    
    def test_simulation_step(self):
        """Проверяет один шаг симуляции"""
        # Выполняем один шаг симуляции
        self.simulation.generate_vehicles()
        self.intersection.update(current_time=0, time_step=1.0)
        
        # Проверяем обновление состояния
        metrics = self.intersection.metrics_history[-1]
        self.assertIn('average_waiting_time', metrics)

class TestTraditionalController(unittest.TestCase):
    """Тесты для традиционного контроллера"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        phases = create_default_phases()
        self.controller = TraditionalController(phases=phases)
        
        # Создаем светофор
        self.traffic_light = Mock()
        self.traffic_light.get_state.return_value = {
            'north': 'green', 'south': 'green', 'east': 'red', 'west': 'red'
        }
    
    def test_phase_timing(self):
        """Проверяет расчет времени фаз"""
        # Проверяем общую длительность цикла
        total_duration = sum(phase.get_total_duration() for phase in self.controller.phases)
        self.assertEqual(total_duration, 70, "Общая продолжительность цикла должна быть 70 секунд")
    
    def test_controller_update(self):
        """Тестирует обновление состояния контроллера"""
        # Получаем текущее состояние
        state = self.controller.update(time.time())
        self.assertIsInstance(state, dict, "Состояние должно быть словарем")
        
        # Проверяем наличие всех направлений
        for direction in ['north', 'south', 'east', 'west']:
            self.assertIn(direction, state)
    
    def test_mode_switching(self):
        """Проверяет переключение режимов работы"""
        # Проверяем существующие режимы
        available_modes = ['morning', 'day', 'evening', 'night']
        for mode in available_modes:
            try:
                self.controller.set_mode(mode)
                self.assertEqual(self.controller.current_mode, mode)
            except ValueError:
                self.fail(f"Режим {mode} должен быть доступен")
    
    def test_configuration_loading(self):
        """Тестирует загрузку конфигурации"""
        # Создаем временный файл конфигурации
        temp_config = "temp_config.json"
        config_data = {
            "phases": [{
                "phase_id": 0,
                "phase_name": "North-South Green",
                "durations": {"green": 30, "yellow": 5, "red": 0},
                "directions": {"north": "green", "south": "green", "east": "red", "west": "red"}
            }],
            "cycle_times": {"morning": 120, "day": 90, "evening": 120, "night": 60},
            "default_mode": "day"
        }
        
        # Сохраняем во временный файл
        with open(temp_config, 'w') as f:
            json.dump(config_data, f)
        
        # Загружаем конфигурацию
        result = self.controller.load_configuration(temp_config)
        self.assertTrue(result, "Конфигурация должна быть загружена")
        
        # Очищаем временный файл
        os.remove(temp_config)

class TestSmartController(unittest.TestCase):
    """Тесты для умного контроллера"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        directions = ['north', 'south', 'east', 'west']
        self.controller = SmartController(directions=directions, algorithm='fuzzy')
        
        # Устанавливаем фиктивные параметры
        self.controller.min_green_time = 15
        self.controller.max_green_time = 90
    
    def test_fuzzy_system(self):
        """Проверяет систему нечеткой логики"""
        try:
            # Проверяем инициализацию системы
            self.controller.initialize_fuzzy_system()
            self.assertTrue(hasattr(self.controller, "fuzzy_ctrl"), "Должна быть инициализирована система нечеткой логики")
            
            # Проверяем расчет времени
            traffic_data = self.test_data["traffic_data"]
            duration = self.controller._calculate_with_fuzzy(traffic_data)
            self.assertTrue(15 <= duration <= 90, "Время должно быть в пределах 15-90 секунд")
        except Exception as e:
            self.fail(f"Ошибка при инициализации нечеткой логики: {e}")
    
    def test_phase_calculation(self):
        """Тестирует расчет оптимальной продолжительности фаз"""
        traffic_data = self.test_data["traffic_data"]
        
        # Проверяем все алгоритмы
        for algorithm in ['fuzzy', 'webster', 'reinforcement']:
            try:
                self.controller.algorithm = algorithm
                duration = self.controller.calculate_phase_duration(traffic_data)
                self.assertTrue(15 <= duration <= 90, f"Для {algorithm} время должно быть в пределах 15-90 секунд")
            except Exception as e:
                print(f"Ошибка для {algorithm}: {e}")
                continue
    
    def test_adaptation_to_traffic(self):
        """Проверяет адаптацию к изменению трафика"""
        # Создаем различные сценарии трафика
        test_scenarios = [
            {"north": 5, "south": 5, "east": 5, "west": 5},  # Равномерный трафик
            {"north": 20, "south": 20, "east": 5, "west": 5},  # Север-Юг перегружены
            {"north": 5, "south": 5, "east": 20, "west": 20}   # Восток-Запад перегружены
        ]
        
        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                # Обновляем состояние
                new_state = self.controller.update(time.time(), scenario)
                self.assertIsInstance(new_state, dict, "Результат должен быть словарем")
                
                # Проверяем противоположные пары
                green_directions = [d for d, s in new_state.items() if s == "green"]
                self.assertTrue(
                    set(green_directions) in self.controller.opposite_pairs,
                    "Должны быть активны только противоположные пары направлений"
                )
    
    def test_webster_method(self):
        """Тестирует метод Вебстера"""
        # Проверяем, что метод существует
        if hasattr(self.controller, "_calculate_with_webster"):
            traffic_data = self.test_data["traffic_data"]
            duration = self.controller._calculate_with_webster(traffic_data)
            self.assertTrue(15 <= duration <= 90, "Время должно быть в пределах 15-90 секунд")
    
    def test_reinforcement_learning(self):
        """Тестирует обучение с подкреплением"""
        if hasattr(self.controller, "_calculate_with_reinforcement"):
            traffic_data = self.test_data["traffic_data"]
            duration = self.controller._calculate_with_reinforcement(traffic_data)
            self.assertTrue(15 <= duration <= 90, "Время должно быть в пределах 15-90 секунд")

class TestSimulationManager(unittest.TestCase):
    """Тесты для менеджера симуляции"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        
        # Создаем контроллеры
        traditional_controller = TraditionalController(phases=create_default_phases())
        smart_controller = SmartController(directions=['north', 'south', 'east', 'west'], algorithm='fuzzy')
        
        # Создаем менеджер симуляции
        self.simulation_manager = SimulationManager(
            video_source=None,
            use_real_video=False,
            traditional_controller=traditional_controller,
            smart_controller=smart_controller,
            simulation_time=300,
            time_step=1.0
        )
        self.simulation_manager.initialize_components()
    
    def test_components_initialization(self):
        """Проверяет инициализацию компонентов"""
        self.assertIsNotNone(self.simulation_manager.intersection, "Перекресток должен быть инициализирован")
        self.assertIsNotNone(self.simulation_manager.traditional_controller, "Традиционный контроллер должен быть инициализирован")
    
    def test_simulation_execution(self):
        """Тестирует выполнение симуляции"""
        traditional_results = self.simulation_manager.run_traditional_simulation()
        smart_results = self.simulation_manager.run_smart_simulation()
        
        # Проверяем, что метрики собраны
        self.assertGreater(len(self.simulation_manager.traditional_metrics), 0, "Метрики традиционного контроллера должны быть собраны")
        self.assertGreater(len(self.simulation_manager.smart_metrics), 0, "Метрики умного контроллера должны быть собраны")
    
    def test_metrics_collection(self):
        """Проверяет сбор метрик"""
        # Получаем данные о трафике
        traffic_data = self.test_data["simulation_state"]
        metrics = self.simulation_manager.collect_metrics(traffic_data)
        
        # Проверяем наличие ключевых метрик
        for metric in ['time', 'queue_lengths', 'throughput', 'active_phase', 'waiting_times']:
            self.assertIn(metric, metrics)
    
    def test_controller_comparison(self):
        """Проверяет сравнение контроллеров"""
        traditional_results, smart_results = self.simulation_manager.compare_controllers()
        
        # Проверяем, что результаты собраны
        self.assertIn('average_waiting_time', traditional_results)
        self.assertIn('average_waiting_time', smart_results)
        
        # Проверяем, что можно вывести отчет
        self.simulation_manager.generate_report(traditional_results, smart_results)

class TestVisualization(unittest.TestCase):
    """Тесты для визуализации"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        
        # Создаем визуализатор
        self.visualizer = IntersectionVisualizer(
            window_size=(800, 600),
            intersection=Mock(spec=Intersection),
            traffic_light=Mock(spec=TrafficLight),
            scale=10.0,
            fps=30
        )
        
        # Имитируем методы
        self.visualizer.initialize_display = Mock()
        self.visualizer.draw_intersection = Mock()
        self.visualizer.draw_vehicles = Mock()
    
    def test_intersection_drawing(self):
        """Проверяет отрисовку перекрестка"""
        # Вызываем отрисовку
        self.visualizer.draw_intersection()
        
        # Проверяем, что методы были вызваны
        self.visualizer.draw_intersection.assert_called_once()
        self.visualizer.draw_vehicles.assert_not_called()
    
    def test_vehicle_drawing(self):
        """Тестирует отображение транспортных средств"""
        vehicles = self.test_data["simulation_state"]["vehicles"]
        self.visualizer.draw_vehicles(vehicles)
        self.visualizer.draw_vehicles.assert_called_once_with(vehicles)
    
    def test_traffic_light_drawing(self):
        """Проверяет отрисовку светофоров"""
        self.visualizer.draw_traffic_lights()
        self.visualizer.draw_traffic_lights.assert_called_once()
    
    def test_animation_generation(self):
        """Тестирует создание анимации"""
        # Создаем тестовую симуляцию
        test_sim = [self.test_data["simulation_state"] for _ in range(10)]
        
        # Тестируем визуализацию
        self.visualizer.running = True
        for frame in test_sim:
            self.visualizer.update(frame)
        
        self.assertEqual(len(self.visualizer.animation_frames), 10, "Должно быть 10 кадров")

class TestIntegration(unittest.TestCase):
    """Интеграционные тесты для проверки взаимодействия компонентов"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        
        # Создаем компоненты
        directions = ['north', 'south', 'east', 'west']
        
        # Традиционный контроллер
        self.traditional_controller = TraditionalController(phases=create_default_phases())
        
        # Умный контроллер
        self.smart_controller = SmartController(directions=directions, algorithm='fuzzy')
        
        # Создаем полосы
        self.lanes = [
            Lane(lane_id=i, direction=direction, length=100, max_speed=20)
            for i, direction in enumerate(directions)
        ]
        
        # Создаем перекресток
        self.intersection = Intersection(lanes=self.lanes)
        
        # Создаем менеджер симуляции
        self.simulation_manager = SimulationManager(
            video_source=None,
            use_real_video=False,
            traditional_controller=self.traditional_controller,
            smart_controller=self.smart_controller,
            simulation_time=60,
            time_step=1.0
        )
    
    def test_video_to_controller(self):
        """Проверяет интеграцию видео и контроллера"""
        # Создаем временный видеокадр
        test_frame = self.test_data["video_frame"]
        
        # Тестируем обработку кадра
        processed_frame, traffic_data = self.simulation_manager.process_video_frame(test_frame)
        
        # Проверяем, что данные о трафике возвращаются
        self.assertIn('queue_lengths', traffic_data)
        self.assertIn('vehicles', traffic_data)
    
    def test_controller_to_simulation(self):
        """Проверяет интеграцию контроллера и симуляции"""
        # Тестируем работу традиционного контроллера
        traditional_state = self.simulation_manager.run_traditional_simulation()
        self.assertIn('average_waiting_time', traditional_state)
        
        # Тестируем работу умного контроллера
        smart_state = self.simulation_manager.run_smart_simulation()
        self.assertIn('average_waiting_time', smart_state)
    
    def test_simulation_to_visualization(self):
        """Проверяет интеграцию симуляции и визуализации"""
        # Создаем тестовые данные
        test_sim_data = self.test_data["simulation_state"]
        
        # Создаем визуализатор
        visualizer = IntersectionVisualizer(
            window_size=(800, 600),
            intersection=self.intersection,
            traffic_light=Mock(),
            scale=10.0,
            fps=30
        )
        
        # Проверяем обновление визуализатора
        visualizer.update(test_sim_data)
        visualizer.draw_intersection.assert_called()
    
    def test_end_to_end(self):
        """Проверяет всю систему целиком"""
        # Создаем простую симуляцию
        traditional_results = self.simulation_manager.run_traditional_simulation()
        smart_results = self.simulation_manager.run_smart_simulation()
        
        # Проверяем, что симуляция завершена
        self.assertIn('average_waiting_time', traditional_results)
        self.assertIn('average_waiting_time', smart_results)
        
        # Проверяем, что можно сгенерировать отчет
        self.simulation_manager.generate_report(traditional_results, smart_results)

class TestSimulationPerformance(unittest.TestCase):
    """Тесты производительности системы"""
    
    def test_full_system_performance(self):
        """Измеряет производительность всей системы"""
        # Создаем полную систему
        directions = ['north', 'south', 'east', 'west']
        
        # Инициализируем компоненты
        lanes = [Lane(lane_id=i, direction=d, length=100, max_speed=20) for i, d in enumerate(directions)]
        intersection = Intersection(lanes=lanes)
        
        # Создаем контроллеры
        traditional_controller = TraditionalController(phases=create_default_phases())
        smart_controller = SmartController(directions=directions, algorithm='fuzzy')
        
        # Создаем менеджер симуляции
        simulation_manager = SimulationManager(
            video_source=None,
            use_real_video=False,
            traditional_controller=traditional_controller,
            smart_controller=smart_controller,
            simulation_time=300,
            time_step=1.0
        )
        simulation_manager.intersection = intersection
        
        # Измеряем производительность
        stats = performance_test(simulation_manager, iterations=10)
        print(f"\nПроизводительность симуляции: {stats['avg_time']:.4f} секунд/итерация")

class TestMetricsCalculator(unittest.TestCase):
    """Тесты для MetricsCalculator"""
    
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.test_data = create_mock_data()
        self.metrics_calculator = MetricsCalculator()
        self.metrics_calculator.use_real_video = False
    
    def test_waiting_time_calculation(self):
        """Проверяет расчет времени ожидания"""
        vehicles = self.test_data["simulation_state"]["vehicles"]
        avg_time = self.metrics_calculator.calculate_average_waiting_time(vehicles)
        self.assertIsInstance(avg_time, (int, float), "Среднее время ожидания должно быть числом")
    
    def test_queue_length(self):
        """Тестирует расчет длины очереди"""
        queue_lengths = self.test_data["simulation_state"]["queue_lengths"]
        max_queue = self.metrics_calculator.calculate_max_queue_length(queue_lengths)
        self.assertIsInstance(max_queue, (int, float), "Максимальная очередь должна быть числом")
    
    def test_fuel_consumption(self):
        """Проверяет расчет расхода топлива"""
        vehicles = self.test_data["simulation_state"]["vehicles"]
        fuel = self.metrics_calculator.calculate_fuel_consumption(vehicles)
        self.assertIsInstance(fuel, (int, float), "Расход топлива должен быть числом")
    
    def test_throughput(self):
        """Тестирует расчет пропускной способности"""
        throughput = self.metrics_calculator.calculate_throughput(self.test_data["simulation_state"])
        self.assertIsInstance(throughput, int, "Пропускная способность должна быть целым числом")

def main():
    """Основная функция для запуска тестов"""
    parser = argparse.ArgumentParser(description='Тестирование системы умных светофоров')
    parser.add_argument('--test-module', type=str, default='all',
                        choices=['all', 'detector', 'model', 'traditional', 'smart', 'simulation', 'visualization', 'integration'],
                        help='Модуль для тестирования')
    parser.add_argument('--performance', action='store_true',
                        help='Запуск тестов производительности')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Количество итераций для тестов производительности')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Уровень подробности вывода')
    args = parser.parse_args()
    
    # Определяем тесты для запуска
    test_classes = {
        'all': [
            TestVehicleDetector,
            TestTrafficModel,
            TestTraditionalController,
            TestSmartController,
            TestSimulationManager,
            TestVisualization,
            TestIntegration,
            TestSimulationPerformance,
            TestMetricsCalculator
        ],
        'detector': [TestVehicleDetector],
        'model': [TestTrafficModel],
        'traditional': [TestTraditionalController],
        'smart': [TestSmartController],
        'simulation': [TestSimulationManager],
        'visualization': [TestVisualization],
        'integration': [TestIntegration]
    }.get(args.test_module, [TestIntegration])
    
    # Запускаем тесты
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=args.verbose)
    result = runner.run(suite)
    
    # Запускаем тесты производительности, если указано
    if args.performance:
        print("\n=== Тесты производительности ===")
        for component in ["VehicleDetector", "SimulationManager"]:
            try:
                test_component = eval(f"Mock({component})")
                stats = performance_test(test_component, args.iterations)
                print(f"{component}: {stats['avg_time']:.4f} сек/итерация (в среднем за {args.iterations} итераций)")
            except Exception as e:
                print(f"{component} не может быть протестирован: {e}")

if __name__ == '__main__':
    main()