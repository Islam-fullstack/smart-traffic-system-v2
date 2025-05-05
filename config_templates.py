import os
import json
import yaml
import argparse

def generate_traditional_controller_config():
    """
    Создает шаблон конфигурации для традиционного контроллера
    """
    return {
        "description": "Конфигурация традиционного контроллера с фиксированными фазами",
        "controllers": {
            "traditional": {
                "description": "Настройки традиционного контроллера светофора",
                "phases": [
                    {
                        "description": "Фаза 1: Север-Юг зеленый, Восток-Запад красный",
                        "phase_id": 0,
                        "phase_name": "North-South Green",
                        "durations": {"green": 30, "yellow": 5, "red": 0},
                        "directions": {"north": "green", "south": "green", "east": "red", "west": "red"}
                    },
                    {
                        "description": "Фаза 2: Восток-Запад зеленый, Север-Юг красный",
                        "phase_id": 1,
                        "phase_name": "East-West Green",
                        "durations": {"green": 30, "yellow": 5, "red": 0},
                        "directions": {"north": "red", "south": "red", "east": "green", "west": "green"}
                    }
                ],
                "cycle_times": {
                    "description": "Длительность цикла в разных режимах",
                    "morning": 120,
                    "day": 90,
                    "evening": 120,
                    "night": 60
                },
                "min_green": {
                    "description": "Минимальное время зеленого сигнала",
                    "default": 15,
                    "unit": "seconds"
                },
                "max_green": {
                    "description": "Максимальное время зеленого сигнала",
                    "default": 90,
                    "unit": "seconds"
                },
                "safety": {
                    "description": "Параметры безопасности и ограничений",
                    "min_yellow": {
                        "description": "Минимальное время желтого сигнала",
                        "default": 5,
                        "unit": "seconds"
                    },
                    "min_red": {
                        "description": "Минимальное время красного сигнала для безопасности",
                        "default": 2,
                        "unit": "seconds"
                    }
                }
            }
        }
    }

def generate_smart_controller_config():
    """
    Создает шаблон конфигурации для умного контроллера
    """
    return {
        "description": "Конфигурация умного адаптивного контроллера",
        "controllers": {
            "smart": {
                "description": "Настройки умного контроллера светофора",
                "algorithm": "fuzzy",
                "learning": {
                    "description": "Параметры обучения и адаптации",
                    "reinforcement": {
                        "description": "Параметры Q-обучения",
                        "learning_rate": 0.1,
                        "discount_factor": 0.95
                    },
                    "fuzzy": {
                        "description": "Нечеткая логика управления",
                        "rules": [
                            "Если очередь большая и трафик интенсивный, тогда увеличить время фазы",
                            "Если очередь маленькая и трафик низкий, тогда уменьшить время фазы"
                        ]
                    }
                },
                "timing": {
                    "description": "Параметры времени сигналов",
                    "min_green": 15,
                    "max_green": 90,
                    "yellow": 5,
                    "unit": "seconds"
                },
                "safety": {
                    "description": "Параметры безопасности",
                    "min_phase_duration": 10,
                    "max_phase_duration": 120
                }
            }
        }
    }

def generate_simulation_config():
    """
    Создает шаблон конфигурации для симуляции
    """
    return {
        "description": "Конфигурация параметров симуляции транспортного потока",
        "simulation": {
            "description": "Основные параметры симуляции",
            "time": 3600,
            "time_step": 1.0,
            "unit": "seconds"
        },
        "intersection": {
            "description": "Параметры модели перекрестка",
            "lanes": ["north", "south", "east", "west"],
            "lane_length": 100,
            "lane_max_speed": 20,
            "unit": "meters"
        },
        "traffic": {
            "description": "Параметры транспортного потока",
            "arrival_rates": {
                "north": 10,
                "south": 10,
                "east": 15,
                "west": 15,
                "unit": "vehicles/hour"
            },
            "vehicle_types": {
                "description": "Распределение типов транспортных средств",
                "car": 0.6,
                "truck": 0.1,
                "bus": 0.2,
                "motorcycle": 0.1
            }
        },
        "random_seed": {
            "description": "Семя генератора случайных чисел",
            "default": 42
        }
    }

def generate_video_processor_config():
    """
    Создает шаблон конфигурации для обработки видео
    """
    return {
        "description": "Конфигурация видеопроцессора",
        "video": {
            "model": {
                "description": "Параметры модели YOLO",
                "path": "yolov8n.pt",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "classes": ["car", "truck", "bus", "motorcycle"]
            },
            "roi": {
                "description": "Области интереса для каждого направления",
                "north": [[100, 200], [300, 400]],
                "south": [[100, 200], [300, 400]],
                "east": [[100, 200], [300, 400]],
                "west": [[100, 200], [300, 400]],
                "unit": "pixels"
            },
            "tracking": {
                "description": "Параметры трекинга объектов",
                "track_buffer": 30,
                "max_age": 15,
                "min_hits": 3,
                "iou_threshold": 0.3
            },
            "output": {
                "description": "Параметры вывода результатов",
                "save_video": False,
                "output_fps": 30
            }
        }
    }

def generate_visualization_config():
    """
    Создает шаблон конфигурации для визуализации
    """
    return {
        "description": "Конфигурация параметров визуализации",
        "visualization": {
            "description": "Параметры визуализации",
            "enabled": True,
            "window_size": "800,600",
            "unit": "pixels",
            "scale": {
                "description": "Масштаб визуализации (пикселей на метр)",
                "default": 10.0
            },
            "fps": {
                "description": "Частота кадров визуализации",
                "default": 30
            },
            "colors": {
                "description": "Цвета для визуализации",
                "background": [255, 255, 255],
                "road": [100, 100, 100],
                "lane": [200, 200, 200],
                "traffic_light": {
                    "red": [255, 0, 0],
                    "yellow": [255, 255, 0],
                    "green": [0, 255, 0]
                },
                "vehicle": {
                    "car": [0, 0, 255],
                    "truck": [128, 0, 255],
                    "bus": [255, 0, 128],
                    "motorcycle": [0, 128, 255]
                }
            },
            "metrics": {
                "description": "Настройки отображения метрик",
                "show_average_waiting_time": True,
                "show_queue_length": True,
                "show_throughput": True,
                "font_size": 24
            }
        }
    }

def generate_main_config():
    """
    Создает объединенную конфигурацию для всей системы
    """
    return {
        "description": "Основная конфигурация системы умных светофоров",
        "mode": {
            "description": "Режим работы системы",
            "values": ["simulation", "video", "comparison"],
            "default": "simulation"
        },
        "components": {
            "description": "Пути к конфигурациям отдельных компонентов",
            "traditional_controller": "configs/traditional_controller.json",
            "smart_controller": "configs/smart_controller.json",
            "video_processor": "configs/video_processor.yaml",
            "simulation": "configs/simulation.json"
        },
        "logging": {
            "description": "Настройки логирования",
            "level": "INFO",
            "file": "logs/system.log",
            "console_output": True
        },
        "features": {
            "description": "Дополнительные функции и флаги",
            "enable_visualization": True,
            "enable_real_time": True,
            "enable_debug_mode": False
        }
    }

def generate_experiment_config():
    """
    Создает шаблон конфигурации для серии экспериментов
    """
    return {
        "description": "Конфигурация серии экспериментов",
        "experiments": [
            {
                "description": "Эксперимент 1: базовая симуляция с традиционным контроллером",
                "name": "traditional_base",
                "type": "simulation",
                "controller": "traditional",
                "parameters": {
                    "simulation_time": 300,
                    "arrival_rates": {"north": 10, "south": 10, "east": 15, "west": 15}
                },
                "metrics": ["average_waiting_time", "max_queue_length", "throughput", "fuel_consumption"]
            },
            {
                "description": "Эксперимент 2: симуляция с нечетким контроллером",
                "name": "fuzzy_1",
                "type": "simulation",
                "controller": "smart",
                "parameters": {
                    "simulation_time": 300,
                    "algorithm": "fuzzy",
                    "arrival_rates": {"north": 10, "south": 10, "east": 15, "west": 15},
                    "min_green": 15,
                    "max_green": 90
                },
                "metrics": ["average_waiting_time", "max_queue_length", "throughput", "fuel_consumption"]
            },
            {
                "description": "Эксперимент 3: симуляция с контроллером на основе Q-обучения",
                "name": "reinforcement_1",
                "type": "simulation",
                "controller": "smart",
                "parameters": {
                    "simulation_time": 300,
                    "algorithm": "reinforcement",
                    "arrival_rates": {"north": 10, "south": 10, "east": 15, "west": 15},
                    "min_green": 15,
                    "max_green": 90
                },
                "metrics": ["average_waiting_time", "max_queue_length", "throughput", "fuel_consumption"]
            }
        ],
        "comparison_metrics": ["average_waiting_time", "max_queue_length", "throughput", "fuel_consumption"],
        "parameter_variations": {
            "description": "Параметры, варьируемые в экспериментах",
            "arrival_rates": [
                {"north": 10, "south": 10, "east": 15, "west": 15},
                {"north": 15, "south": 15, "east": 20, "west": 20}
            ],
            "algorithms": ["fuzzy", "reinforcement", "webster"]
        }
    }

def save_config_as_json(config, filename):
    """
    Сохраняет конфигурацию в формате JSON
    """
    try:
        with open(filename + ".json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ошибка сохранения JSON: {e}")
        return False

def save_config_as_yaml(config, filename):
    """
    Сохраняет конфигурацию в формате YAML
    """
    try:
        with open(filename + ".yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Ошибка сохранения YAML: {e}")
        return False

def main():
    """
    Основная функция для генерации шаблонов конфигураций
    """
    parser = argparse.ArgumentParser(description='Генератор шаблонов конфигураций')
    parser.add_argument('--output-dir', type=str, default='configs', help='Директория для сохранения конфигураций')
    parser.add_argument('--format', type=str, default='both', choices=['json', 'yaml', 'both'], help='Формат конфигурации')
    parser.add_argument('--template', type=str, default='all', 
                        choices=['all', 'traditional', 'smart', 'simulation', 'video', 'visualization', 'main', 'experiment'],
                        help='Тип шаблона для генерации')
    args = parser.parse_args()

    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Генерируем и сохраняем шаблоны
    templates = []
    
    if args.template == 'all' or args.template == 'traditional':
        templates.append(('traditional_controller', generate_traditional_controller_config()))
    if args.template == 'all' or args.template == 'smart':
        templates.append(('smart_controller', generate_smart_controller_config()))
    if args.template == 'all' or args.template == 'simulation':
        templates.append(('simulation', generate_simulation_config()))
    if args.template == 'all' or args.template == 'video':
        templates.append(('video_processor', generate_video_processor_config()))
    if args.template == 'all' or args.template == 'visualization':
        templates.append(('visualization', generate_visualization_config()))
    if args.template == 'all' or args.template == 'main':
        templates.append(('main', generate_main_config()))
    if args.template == 'all' or args.template == 'experiment':
        templates.append(('experiment', generate_experiment_config()))

    # Сохраняем конфигурации
    generated_files = []
    
    for name, config in templates:
        if args.format in ['json', 'both']:
            if save_config_as_json(config, os.path.join(args.output_dir, name)):
                generated_files.append(os.path.join(args.output_dir, name + ".json"))
                
        if args.format in ['yaml', 'both']:
            if save_config_as_yaml(config, os.path.join(args.output_dir, name)):
                generated_files.append(os.path.join(args.output_dir, name + ".yaml"))
    
    # Выводим информацию о созданных файлах
    print(f"\nСоздано {len(generated_files)} файлов:")
    for file in generated_files:
        print(f"- {file}")

if __name__ == "__main__":
    main()