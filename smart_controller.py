import numpy as np
import json
import time
import argparse  # ✅ Добавлено

from enum import Enum

class TrafficLightColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"

class SmartController:
    """Интеллектуальный адаптивный контроллер светофора"""
    def __init__(self, directions, min_green_time=15, max_green_time=90,
                 yellow_time=5, algorithm='fuzzy'):
        # Список всех направлений
        self.directions = directions
        
        # Противоположные пары направлений
        self.opposite_pairs = [
            ['north', 'south'],
            ['east', 'west']
        ]
        
        # Текущая активная пара направлений
        self.active_pair_index = 0
        self.active_directions = self.opposite_pairs[self.active_pair_index]
        
        # Настройки времени
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.algorithm = algorithm
        
        # Время начала работы
        self.start_time = None
        self.phase_timer = 0

    def start(self):
        """Запускает работу контроллера и инициализирует таймер"""
        self.start_time = time.time()
        self.phase_timer = 0

    def update(self, current_time, traffic_data):
        """Обновляет состояние светофора на основе данных о трафике"""
        if self.start_time is None:
            self.start()
        
        # Обновляем таймер
        self.phase_timer = current_time - self.start_time
        
        # Проверяем, завершена ли текущая фаза
        if self._is_phase_complete(traffic_data):
            self.switch_phase(traffic_data)
        
        # Возвращаем текущее состояние
        return self.get_current_state()

    def _is_phase_complete(self, traffic_data):
        """Проверяет, завершена ли текущая фаза"""
        # Рассчитываем время для активной пары
        current_pair = self.opposite_pairs[self.active_pair_index]
        
        # Получаем данные о трафике для активной пары
        pair_traffic = {d: traffic_data.get(d, 0) for d in current_pair}
        
        # Рассчитываем оптимальное время
        required_duration = self.calculate_phase_duration(pair_traffic)
        
        return self.phase_timer >= required_duration

    def calculate_phase_duration(self, traffic_data):
        """Вычисляет оптимальную продолжительность фазы"""
        if self.algorithm == 'fuzzy':
            return self._calculate_with_fuzzy(traffic_data)
        elif self.algorithm == 'webster':
            return self._calculate_with_webster(traffic_data)
        elif self.algorithm == 'reinforcement':
            return self._calculate_with_reinforcement(traffic_data)
        return self.min_green_time  # По умолчанию

    def _calculate_with_fuzzy(self, traffic_data):
        """Рассчитывает продолжительность с помощью нечеткой логики"""
        # Пример реализации
        queue_values = list(traffic_data.values())
        avg_queue = np.mean(queue_values)
        
        # Базовый расчет на основе очереди
        base_time = self.min_green_time + (avg_queue / 20) * \
                   (self.max_green_time - self.min_green_time)
        
        return np.clip(base_time, self.min_green_time, self.max_green_time)

    def _calculate_with_webster(self, traffic_data):
        """Рассчитывает продолжительность по методу Вебстера"""
        # Пример упрощенного расчета
        total_vehicles = sum(traffic_data.values())
        optimal_duration = 60 + total_vehicles * 2
        return np.clip(optimal_duration, self.min_green_time, self.max_green_time)

    def _calculate_with_reinforcement(self, traffic_data):
        """Рассчитывает продолжительность с помощью Q-обучения"""
        # Пример упрощенного расчета
        total_vehicles = sum(traffic_data.values())
        return 30 + total_vehicles * 1.5

    def switch_phase(self, traffic_data=None):
        """Переключает текущую активную фазу на противоположную пару"""
        # Сначала желтый сигнал для всех активных направлений
        # Это должно быть реализовано в визуализаторе
        
        # Ждем желтый сигнал
        time.sleep(self.yellow_time)
        
        # Переключаем на следующую пару направлений
        self.active_pair_index = (self.active_pair_index + 1) % len(self.opposite_pairs)
        self.active_directions = self.opposite_pairs[self.active_pair_index]
        
        # Сбрасываем таймер
        self.phase_timer = 0
        self.start_time = time.time()

    def get_current_state(self):
        """Возвращает текущее состояние всех направлений"""
        current_state = {}
        for direction in self.directions:
            if direction in self.active_directions:
                current_state[direction] = "green"
            else:
                current_state[direction] = "red"
        return current_state

    def save_state(self, state_file):
        """Сохраняет текущее состояние в файл"""
        try:
            state = {
                'active_directions': self.active_directions,
                'phase_timer': self.phase_timer,
                'start_time': self.start_time,
                'algorithm': self.algorithm,
                'min_green_time': self.min_green_time,
                'max_green_time': self.max_green_time,
                'yellow_time': self.yellow_time
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения состояния: {e}")
            return False

    def load_configuration(self, config_file):
        """Загружает конфигурацию из JSON файла"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            self.active_directions = config.get('active_directions', ['north'])
            self.min_green_time = config.get('min_green_time', self.min_green_time)
            self.max_green_time = config.get('max_green_time', self.max_green_time)
            self.yellow_time = config.get('yellow_time', self.yellow_time)
            self.algorithm = config.get('algorithm', self.algorithm)
            
            # Обновляем индекс активной пары
            if self.active_directions in self.opposite_pairs:
                self.active_pair_index = self.opposite_pairs.index(self.active_directions)
            else:
                self.active_pair_index = 0
                self.active_directions = self.opposite_pairs[self.active_pair_index]
                
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return False

def main():
    """Основная функция для запуска контроллера"""
    parser = argparse.ArgumentParser(description='Умный контроллер светофора')
    parser.add_argument('--algorithm', type=str, default='fuzzy', 
                        choices=['fuzzy', 'reinforcement', 'webster'],
                        help='Алгоритм адаптации')
    parser.add_argument('--min-green', type=int, default=15,
                        help='Минимальное время зеленого сигнала в секундах')
    parser.add_argument('--max-green', type=int, default=90,
                        help='Максимальное время зеленого сигнала в секундах')
    parser.add_argument('--simulation-time', type=int, default=300,
                        help='Время симуляции в секундах')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    args = parser.parse_args()

    # Создаем контроллер
    directions = ['north', 'south', 'east', 'west']
    controller = SmartController(
        directions=directions,
        min_green_time=args.min_green,
        max_green_time=args.max_green,
        algorithm=args.algorithm
    )

    # Загружаем конфигурацию, если указана
    if args.config:
        controller.load_configuration(args.config)

    # Симулируем работу контроллера
    print(f"\nЗапуск симуляции с алгоритмом {args.algorithm}...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.simulation_time:
            # Генерируем имитационные данные о трафике
            traffic_data = {
                'north': np.random.randint(0, 15),
                'south': np.random.randint(0, 15),
                'east': np.random.randint(0, 15),
                'west': np.random.randint(0, 15)
            }
            
            # Обновляем состояние контроллера
            current_state = controller.update(time.time(), traffic_data)
            
            # Выводим информацию
            print(f"\nВремя: {int(time.time() - start_time)} сек")
            print(f"Данные о трафике: {traffic_data}")
            print(f"Текущее состояние: {current_state}")
            print(f"Активные направления: {controller.active_directions}")
            
            # Ждем 5 секунд перед следующим обновлением
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nСимуляция остановлена пользователем")
    
    print("\nСимуляция завершена")

if __name__ == "__main__":
    main()