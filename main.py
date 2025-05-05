import os
import sys
import time
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import pygame
from argparse import ArgumentParser
import logging

# Импортируем разработанные модули
try:
    from video_processor import VehicleDetector
    from traffic_model import Intersection, Lane, Vehicle, TrafficSimulation
    from traditional_controller import TraditionalController, create_default_phases
    from smart_controller import SmartController
    from simulation_manager import SimulationManager
    from visualization import IntersectionVisualizer, MetricsPlotter, DashboardUI
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Пожалуйста, убедитесь, что все модули доступны в PYTHONPATH")
    sys.exit(1)

class ConfigurationManager:
    """Класс для работы с конфигурацией системы"""
    def __init__(self, config_file=None, default_config=None):
        self.config_file = config_file
        self.default_config = default_config or self._default_configuration()
        self.config = self.load_config()

    def _default_configuration(self):
        """Возвращает конфигурацию по умолчанию"""
        return {
            "simulation": {
                "time": 300,
                "time_step": 1.0,
                "lanes": ["north", "south", "east", "west"],
                "arrival_rates": {"north": 10, "south": 10, "east": 15, "west": 15},
                "vehicle_types": {"car": 0.6, "truck": 0.1, "bus": 0.2, "motorcycle": 0.1}
            },
            "controllers": {
                "traditional": {"type": "fixed"},
                "smart": {"algorithm": "fuzzy"}
            },
            "visualization": {
                "window_size": "800,600",
                "scale": 10.0,
                "fps": 30
            }
        }

    def load_config(self):
        """Загружает конфигурацию из файла"""
        if not self.config_file or not os.path.exists(self.config_file):
            logging.info("Используется конфигурация по умолчанию")
            return self.default_config
        
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Ошибка загрузки конфигурации: {e}")
            return self.default_config

    def save_config(self, config_data, output_file=None):
        """Сохраняет конфигурацию в файл"""
        output_file = output_file or self.config_file or "config.json"
        try:
            with open(output_file, 'w') as f:
                if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Не удалось сохранить конфигурацию: {e}")
            return False

    def get_value(self, key, default=None):
        """Получает значение из конфигурации"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update_value(self, key, value):
        """Обновляет значение в конфигурации"""
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    def get_section(self, section_name):
        """Возвращает секцию конфигурации"""
        return self.config.get(section_name, {})

class SmartTrafficSystem:
    """Основной класс для умной системы управления светофорами"""
    def __init__(self, config_file=None, mode="simulation", output_dir="results"):
        self.config_manager = ConfigurationManager(config_file)
        self.mode = mode
        self.output_dir = output_dir
        self.simulation_manager = None
        self.traditional_controller = None
        self.smart_controller = None
        self.intersection = None
        self.vehicle_detector = None
        self.visualizer = None
        self.metrics_plotter = None

    def load_configuration(self):
        """Загружает и применяет конфигурацию"""
        # Загружаем конфигурацию
        config = self.config_manager.config
        
        # Загружаем параметры симуляции
        sim_config = config.get("simulation", {})
        self.simulation_time = sim_config.get("time", 300)
        self.time_step = sim_config.get("time_step", 1.0)
        
        # Загружаем контроллеры
        controller_config = config.get("controllers", {})
        self.traditional_config = controller_config.get("traditional", {})
        self.smart_config = controller_config.get("smart", {})
        
        # Параметры визуализации
        vis_config = config.get("visualization", {})
        self.window_size = tuple(map(int, vis_config.get("window_size", "800,600").split(',')))
        self.scale = vis_config.get("scale", 10.0)
        self.fps = vis_config.get("fps", 30)

    def initialize_components(self):
        """Инициализирует все компоненты системы"""
        # Получаем направления из конфигурации
        directions = self.config_manager.get_value("simulation.lanes", ["north", "south", "east", "west"])
        
        # Инициализируем контроллеры
        traditional_config = self.config_manager.get_section("controllers.traditional")
        smart_config = self.config_manager.get_section("controllers.smart")
        
        # Создаем традиционный контроллер
        phases = create_default_phases()
        if traditional_config.get("use_custom_phases", False) and traditional_config.get("phases_file"):
            traditional_controller = TraditionalController(phases=[])
            traditional_controller.load_configuration(traditional_config["phases_file"])
        else:
            traditional_controller = TraditionalController(phases=phases)
        
        # Создаем умный контроллер
        smart_controller = SmartController(
            directions=directions,
            algorithm=smart_config.get("algorithm", "fuzzy"),
            min_green_time=smart_config.get("min_green", 15),
            max_green_time=smart_config.get("max_green", 90),
            yellow_time=smart_config.get("yellow", 5)
        )
        
        # Сохраняем контроллеры
        self.traditional_controller = traditional_controller
        self.smart_controller = smart_controller
        
        # Инициализируем симуляцию
        if self.mode in ["simulation", "comparison"]:
            # Создаем полосы движения
            lanes = [
                Lane(lane_id=i, direction=direction, length=100, max_speed=20)
                for i, direction in enumerate(directions)
            ]
            
            # Создаем перекресток
            self.intersection = Intersection(lanes=lanes)
            
            # Инициализируем менеджер симуляции
            self.simulation_manager = SimulationManager(
                video_source=None,
                use_real_video=False,
                traditional_controller=traditional_controller,
                smart_controller=smart_controller,
                simulation_time=self.simulation_time,
                time_step=self.time_step
            )
            
            # Инициализируем визуализатор
            if self.config_manager.get_value("visualization.enabled", True):
                self.visualizer = IntersectionVisualizer(
                    window_size=self.window_size,
                    intersection=self.intersection,
                    traffic_light=TrafficLight(self.traditional_controller),
                    scale=self.scale,
                    fps=self.fps,
                    background_color=(255, 255, 255)
                )
                self.metrics_plotter = MetricsPlotter(
                    traditional_data=[],
                    smart_data=[],
                    output_dir=self.output_dir
                )

    def run_simulation_mode(self):
        """Запускает симуляцию транспортного потока"""
        logging.info("Запуск симуляции...")
        self.simulation_manager.initialize_components()
        traditional_results, smart_results = self.simulation_manager.compare_controllers()
        logging.info("Симуляция завершена")
        return traditional_results, smart_results

    def run_video_mode(self):
        """Запускает обработку видео"""
        logging.info("Запуск обработки видео...")
        self.simulation_manager.use_real_video = True
        self.simulation_manager.video_source = self.config_manager.get_value("video.source", 0)
        self.simulation_manager.initialize_components()
        
        # Запускаем симуляцию с реальным видео
        traditional_results = self.simulation_manager.run_traditional_simulation()
        smart_results = self.simulation_manager.run_smart_simulation()
        logging.info("Обработка видео завершена")
        return traditional_results, smart_results

    def run_comparison_mode(self):
        """Запускает сравнение контроллеров"""
        logging.info("Запуск сравнения контроллеров...")
        traditional_results, smart_results = self.simulation_manager.compare_controllers()
        logging.info("Сравнение контроллеров завершено")
        return traditional_results, smart_results

    def visualize_results(self):
        """Визуализирует результаты симуляции"""
        if self.config_manager.get_value("visualization.enabled", True):
            if self.visualizer and self.metrics_plotter:
                # Визуализируем симуляцию
                test_data = [{
                    'queue_lengths': {d: np.random.randint(0, 10) for d in self.intersection.lanes},
                    'vehicles': self.intersection.lanes['north'].vehicles,
                    'active_phase': 'North-South Green',
                    'throughput': np.random.randint(0, 20),
                    'average_waiting_time': np.random.uniform(0, 60)
                } for _ in range(100)]
                
                # Рисуем анимацию
                self.visualizer.run_animation(test_data)
                
                # Рисуем графики
                self.metrics_plotter.traditional_data = self.simulation_manager.traditional_metrics
                self.metrics_plotter.smart_data = self.simulation_manager.smart_metrics
                self.metrics_plotter.save_all_plots()
                logging.info(f"Результаты визуализированы и сохранены в {self.output_dir}")

    def save_results(self, traditional_results, smart_results):
        """Сохраняет результаты симуляции"""
        results = {
            "traditional": traditional_results,
            "smart": smart_results,
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode
        }
        
        # Сохраняем в JSON
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"results_{self.mode}_{int(time.time())}.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logging.info(f"Результаты сохранены в {output_file}")
            return True
        except Exception as e:
            logging.error(f"Не удалось сохранить результаты: {e}")
            return False

    def run(self):
        """Запускает систему в выбранном режиме"""
        # Загружаем конфигурацию
        self.load_configuration()
        
        # Инициализируем компоненты
        self.initialize_components()
        
        # Запускаем в нужном режиме
        if self.mode == "simulation":
            traditional_results, smart_results = self.run_simulation_mode()
        elif self.mode == "video":
            traditional_results, smart_results = self.run_video_mode()
        elif self.mode == "comparison":
            traditional_results, smart_results = self.run_comparison_mode()
        else:
            logging.error(f"Неизвестный режим: {self.mode}")
            return False
        
        # Визуализируем результаты
        self.visualize_results()
        
        # Сохраняем результаты
        self.save_results(traditional_results, smart_results)
        
        # Выводим отчет
        self.generate_report(traditional_results, smart_results)
        return True

    def generate_report(self, traditional_results, smart_results):
        """Генерирует сводный отчет по результатам"""
        print("\n=== Сводный отчет ===")
        print("\nТрадиционный контроллер:")
        for metric, value in traditional_results.items():
            print(f"- {metric}: {value:.2f}")
        
        print("\nУмный контроллер:")
        for metric, value in smart_results.items():
            print(f"- {metric}: {value:.2f}")
        
        print("\nРекомендации:")
        if smart_results.get('average_waiting_time', float('inf')) < traditional_results.get('average_waiting_time', float('inf')):
            print("- Умный контроллер показывает лучшие результаты по времени ожидания")
        if smart_results.get('max_queue_length', float('inf')) < traditional_results.get('max_queue_length', float('inf')):
            print("- Умный контроллер лучше управляет очередями")
        if smart_results.get('throughput', 0) > traditional_results.get('throughput', 0):
            print("- Умный контроллер демонстрирует более высокую пропускную способность")

class ExperimentRunner:
    """Класс для запуска серии экспериментов"""
    def __init__(self, system, experiment_configs, output_dir="results"):
        self.system = system
        self.experiment_configs = experiment_configs
        self.output_dir = output_dir
        self.experiment_results = []
    
    def run_experiment(self, experiment_config):
        """Запускает один эксперимент"""
        logging.info(f"Запуск эксперимента с конфигурацией: {experiment_config}")
        self.system.config_manager.update_value("simulation.time", experiment_config.get("simulation_time", 300))
        self.system.config_manager.update_value("controllers.smart.algorithm", experiment_config.get("algorithm", "fuzzy"))
        
        # Сохраняем текущие настройки
        config_file = os.path.join(self.output_dir, f"config_{int(time.time())}.json")
        self.system.config_manager.save_config(self.system.config_manager.config, config_file)
        
        # Запускаем систему
        traditional_results, smart_results = self.system.run()
        result = {
            "config": experiment_config,
            "traditional": traditional_results,
            "smart": smart_results
        }
        self.experiment_results.append(result)
        
        # Сохраняем результаты
        result_file = os.path.join(self.output_dir, f"result_{experiment_config.get('name', 'unnamed')}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def run_all_experiments(self):
        """Запускает все эксперименты"""
        logging.info(f"Запуск серии экспериментов ({len(self.experiment_configs)} конфигураций)")
        progress_bar = tqdm(self.experiment_configs)
        
        for i, config in enumerate(progress_bar):
            progress_bar.set_description(f"Эксперимент {i+1}/{len(self.experiment_configs)}")
            self.run_experiment(config)
        
        logging.info("Все эксперименты завершены")
        self.generate_report()
    
    def generate_report(self):
        """Генерирует сводный отчет по всем экспериментам"""
        plt.figure(figsize=(12, 6))
        
        # Время ожидания
        trad_wait_times = [r['traditional']['average_waiting_time'] for r in self.experiment_results]
        smart_wait_times = [r['smart']['average_waiting_time'] for r in self.experiment_results]
        x = list(range(len(self.experiment_results)))
        
        plt.plot(x, trad_wait_times, label='Традиционный контроллер', marker='o')
        plt.plot(x, smart_wait_times, label='Умный контроллер', marker='s')
        plt.xlabel('Эксперимент')
        plt.ylabel('Среднее время ожидания')
        plt.title('Сравнение алгоритмов управления светофорами')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'experiment_comparison.png'))
        plt.show()
        
        # Сохраняем сводный отчет
        summary = {
            "experiments": self.experiment_results,
            "best_waiting_time": min(self.experiment_results, key=lambda r: r['smart']['average_waiting_time']),
            "best_queue": min(self.experiment_results, key=lambda r: r['smart']['max_queue_length'])
        }
        
        report_file = os.path.join(self.output_dir, "experiment_summary.json")
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Сводный отчет сохранен в {report_file}")

def parse_arguments():
    """Разбирает аргументы командной строки"""
    parser = argparse.ArgumentParser(description='Умная система управления светофорами')
    
    # Основные параметры
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='simulation', 
                        choices=['simulation', 'video', 'comparison'],
                        help='Режим работы системы')
    parser.add_argument('--output-dir', type=str, default='results', 
                        help='Директория для сохранения результатов')
    
    # Параметры видео
    parser.add_argument('--video-source', type=str, default='0', 
                        help='Путь к видео или индекс камеры')
    
    # Параметры симуляции
    parser.add_argument('--simulation-time', type=int, default=300, 
                        help='Время симуляции в секундах')
    parser.add_argument('--traditional-controller', type=str, default='fixed',
                        help='Тип традиционного контроллера')
    parser.add_argument('--smart-controller', type=str, default='fuzzy',
                        help='Тип умного контроллера')
    
    # Дополнительные параметры
    parser.add_argument('--visualize', action='store_true', help='Включить визуализацию')
    parser.add_argument('--experiment', type=str, help='Путь к файлу конфигурации эксперимента')
    parser.add_argument('--debug', action='store_true', help='Включить отладочный режим')
    
    args = parser.parse_args()
    
    # Проверяем совместимость аргументов
    if args.mode == 'video' and not args.video_source:
        parser.error("Для режима video требуется указать --video-source")
    
    if args.mode in ['simulation', 'comparison'] and args.simulation_time <= 0:
        parser.error("Время симуляции должно быть положительным числом")
    
    return args

def check_dependencies():
    """Проверяет наличие всех зависимостей"""
    dependencies = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'cv2': 'opencv-python',
        'skfuzzy': 'scikit-fuzzy',
        'yaml': 'pyyaml',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
        'pygame': 'pygame',
        'argparse': 'argparse',
        'logging': 'logging'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            if package:
                missing.append(package)
    
    if missing:
        print(f"Отсутствуют зависимости: {', '.join(missing)}")
        print("Пожалуйста, установите их с помощью pip install:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def text_interface():
    """Простой текстовый интерфейс для выбора режима"""
    print("=== Умная система управления светофорами ===")
    print("Выберите режим работы:")
    print("1 - Симуляция с трафиком")
    print("2 - Обработка видео")
    print("3 - Сравнение контроллеров")
    print("4 - Запуск серии экспериментов")
    
    choice = input("Введите номер режима (1-4): ")
    modes = {
        "1": "simulation",
        "2": "video",
        "3": "comparison",
        "4": "experiment"
    }
    
    return modes.get(choice, "simulation")

def main():
    """Основная функция"""
    # Проверяем зависимости
    if not check_dependencies():
        return
    
    # Разбираем аргументы командной строки
    args = parse_arguments()
    
    # Инициализируем логирование
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Если не указаны аргументы, запускаем текстовый интерфейс
    if len(sys.argv) == 1:
        args.mode = text_interface()
    
    # Создаем систему
    traffic_system = SmartTrafficSystem(
        config_file=args.config,
        mode=args.mode,
        output_dir=args.output_dir
    )
    
    # Инициализируем компоненты
    traffic_system.load_configuration()
    
    # Запускаем в зависимости от режима
    if args.mode == 'experiment':
        # Загружаем конфигурации экспериментов
        if not args.experiment or not os.path.exists(args.experiment):
            print("Файл конфигурации экспериментов не найден")
            return
        
        with open(args.experiment, 'r') as f:
            if args.experiment.endswith('.yaml') or args.experiment.endswith('.yml'):
                experiment_configs = yaml.safe_load(f)
            else:
                experiment_configs = json.load(f)
        
        # Создаем runner и запускаем эксперименты
        experiment_runner = ExperimentRunner(
            system=traffic_system,
            experiment_configs=experiment_configs,
            output_dir=args.output_dir
        )
        experiment_runner.run_all_experiments()
    else:
        # Запускаем одиночный режим
        traditional_results, smart_results = None, None
        
        if args.mode == 'simulation':
            traditional_results, smart_results = traffic_system.run_simulation_mode()
        elif args.mode == 'video':
            traditional_results, smart_results = traffic_system.run_video_mode()
        elif args.mode == 'comparison':
            traditional_results, smart_results = traffic_system.run_comparison_mode()
        
        # Визуализируем результаты
        if args.visualize:
            traffic_system.visualize_results()
        
        # Сохраняем результаты
        traffic_system.save_results(traditional_results, smart_results)

if __name__ == "__main__":
    main()