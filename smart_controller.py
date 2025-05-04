import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный backend

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
import argparse
import time
import matplotlib.pyplot as plt

class SmartController:
    """Интеллектуальный адаптивный контроллер светофора"""
    def __init__(self, directions, min_green_time=15, max_green_time=90, 
             yellow_time=5, algorithm='fuzzy'):
        self.directions = directions
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.yellow_time = yellow_time
        self.algorithm = algorithm

        # Инициализация состояния светофора
        self.current_phase = {direction: 'red' for direction in directions}
        self.active_directions = []
        self.phase_timer = 0
        self.start_time = None
        self.fuzzy_system = None
        self.fuzzy_control_system = None  # <-- Новый атрибут
        self.fuzzy_vars = {}             # <-- Новый атрибут
        self.webster_method = None
        self.rl_controller = None
        
        # Инициализация системы нечеткой логики
        if algorithm == 'fuzzy':
            self.initialize_fuzzy_system()
        elif algorithm == 'webster':
            self.webster_method = WebsterMethod()
        elif algorithm == 'reinforcement':
            self.rl_controller = ReinforcementLearningController()

    def start(self):
        """Запускает работу контроллера и инициализирует таймер"""
        self.start_time = time.time()
        self.phase_timer = 0
        # Начинаем с первого направления
        self.active_directions = [self.directions[0]]
        self._update_phase_state()

    def update(self, current_time, traffic_data):
        """Обновляет состояние светофора на основе данных о трафике"""
        if self.start_time is None:
            self.start()
        
        # Обновляем таймер
        self.phase_timer = current_time - (self.start_time or current_time)
        
        # Проверяем, нужно ли переключить фазу
        if self._is_phase_complete(traffic_data):
            self.switch_phase(traffic_data)
        
        return self.get_current_state()

    def initialize_fuzzy_system(self):
        """Инициализирует систему нечеткой логики"""
        try:
            # Определяем диапазоны входных и выходных переменных
            queue_length = ctrl.Antecedent(np.arange(0, 101, 1), 'queue_length')
            waiting_time = ctrl.Antecedent(np.arange(0, 121, 1), 'waiting_time')
            green_time = ctrl.Consequent(np.arange(0, 101, 1), 'green_time')

            # Автоматически создаем функции принадлежности
            queue_length.automf(3, names=['low', 'medium', 'high'])
            waiting_time.automf(3, names=['short', 'moderate', 'long'])
            green_time.automf(3, names=['short', 'medium', 'long'])

            # Создаем правила нечеткой логики
            rules = [
                ctrl.Rule(queue_length['low'] & waiting_time['short'], green_time['short']),
                ctrl.Rule(queue_length['low'] & waiting_time['moderate'], green_time['short']),
                ctrl.Rule(queue_length['low'] & waiting_time['long'], green_time['medium']),
                ctrl.Rule(queue_length['medium'] & waiting_time['short'], green_time['medium']),
                ctrl.Rule(queue_length['medium'] & waiting_time['moderate'], green_time['medium']),
                ctrl.Rule(queue_length['medium'] & waiting_time['long'], green_time['long']),
                ctrl.Rule(queue_length['high'] & waiting_time['short'], green_time['medium']),
                ctrl.Rule(queue_length['high'] & waiting_time['moderate'], green_time['long']),
                ctrl.Rule(queue_length['high'] & waiting_time['long'], green_time['long'])
            ]

            # Сохраняем ControlSystem и переменные
            self.fuzzy_control_system = ctrl.ControlSystem(rules)
            self.fuzzy_vars['queue_length'] = queue_length
            self.fuzzy_vars['waiting_time'] = waiting_time
            self.fuzzy_vars['green_time'] = green_time

            # Создаем систему управления
            self.fuzzy_system = ctrl.ControlSystemSimulation(self.fuzzy_control_system)

            # Визуализируем функции принадлежности
            self._visualize_fuzzy_system()

        except Exception as e:
            print(f"Предупреждение: Ошибка инициализации нечеткой системы - {e}")

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
        """Рассчитывает продолжительность фазы с помощью нечеткой логики"""
        if not self.fuzzy_system:
            return self.min_green_time
            
        # Получаем средние значения очереди и времени ожидания
        queue_values = [traffic_data.get(d, 0) for d in self.directions]
        avg_queue = np.mean(queue_values)
        avg_waiting = np.mean([max(0, 10 * (q > 0)) for q in queue_values])  # Пример
        
        # Ограничиваем значения
        avg_queue = np.clip(avg_queue, 0, 100)
        avg_waiting = np.clip(avg_waiting, 0, 120)
        
        # Рассчитываем
        self.fuzzy_system.input['queue_length'] = avg_queue
        self.fuzzy_system.input['waiting_time'] = avg_waiting
        self.fuzzy_system.compute()
        
        # Преобразуем в реальное время
        fuzzy_time = self.fuzzy_system.output['green_time']
        real_time = self.min_green_time + (fuzzy_time/100) * \
                   (self.max_green_time - self.min_green_time)
        
        return np.clip(real_time, self.min_green_time, self.max_green_time)

    def _calculate_with_webster(self, traffic_data):
        """Рассчитывает продолжительность фазы с помощью метода Вебстера"""
        if not self.webster_method:
            return self.min_green_time
            
        # Преобразуем данные о трафике в объемы движения
        volumes = [traffic_data.get(d, 0) for d in self.directions]
        volumes = [v * 60 for v in volumes]  # Преобразуем в автомобили в час
        
        # Рассчитываем оптимальный цикл
        cycle_length = self.webster_method.calculate_optimal_cycle(volumes)
        green_times = self.webster_method.calculate_green_splits(volumes, cycle_length)
        
        # Возвращаем максимальное время зеленого
        return np.max(green_times) if green_times else self.min_green_time

    def _calculate_with_reinforcement(self, traffic_data):
        """Рассчитывает продолжительность фазы с помощью Q-обучения"""
        if not self.rl_controller:
            return self.min_green_time
            
        # Получаем состояние
        state = self.rl_controller.get_state_index(traffic_data)
        
        # Выбираем действие
        action = self.rl_controller.choose_action(state)
        
        # Применяем действие (преобразуем в реальное время)
        real_time = self.min_green_time + (action/10) * \
                   (self.max_green_time - self.min_green_time)
        
        # Обновляем Q-таблицу (в реальном сценарии здесь будет награда)
        next_state = self.rl_controller.get_state_index(self._generate_next_traffic(traffic_data))
        self.rl_controller.update_q_table(state, action, 0, next_state)
        
        return real_time

    def _generate_next_traffic(self, traffic_data):
        """Генерирует следующие данные о трафике для Q-обучения"""
        return {d: max(0, v + np.random.randint(-2, 5)) 
                for d, v in traffic_data.items()}

    def _is_phase_complete(self, traffic_data):
        """Проверяет, завершена ли текущая фаза"""
        # Рассчитываем нужное время
        required_duration = self.calculate_phase_duration(traffic_data)
        
        # Проверяем, прошло ли достаточно времени
        return self.phase_timer >= required_duration

    def switch_phase(self, traffic_data=None):
        """Переключает текущую активную фазу"""
        # Сначала желтый сигнал
        self._set_phase('yellow')
        time.sleep(self.yellow_time)
        
        # Переключаем направления
        current_idx = self.directions.index(self.active_directions[0]) if self.active_directions else 0
        next_idx = (current_idx + 1) % len(self.directions)
        self.active_directions = [self.directions[next_idx]]
        
        # Обновляем таймер
        self.phase_timer = 0
        self.start_time = time.time()
        
        # Устанавливаем новую фазу
        self._update_phase_state()

    def _update_phase_state(self):
        """Обновляет состояние светофора на основе активных направлений"""
        new_state = {}
        for direction in self.directions:
            if direction in self.active_directions:
                new_state[direction] = 'green'
            else:
                new_state[direction] = 'red'
        self.current_phase = new_state

    def _set_phase(self, color):
        """Устанавливает указанную фазу для активных направлений"""
        new_state = self.current_phase.copy()
        for direction in self.active_directions:
            new_state[direction] = color
        self.current_phase = new_state

    def get_current_state(self):
        """Возвращает текущее состояние всех направлений"""
        return self.current_phase.copy()

    def save_state(self, state_file):
        """Сохраняет текущее состояние в файл"""
        try:
            state = {
                'directions': self.directions,
                'min_green_time': self.min_green_time,
                'max_green_time': self.max_green_time,
                'yellow_time': self.yellow_time,
                'algorithm': self.algorithm,
                'current_phase': self.current_phase,
                'active_directions': self.active_directions,
                'phase_timer': self.phase_timer,
                'start_time': self.start_time
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
                
            # Обновляем параметры
            self.directions = config.get('directions', self.directions)
            self.min_green_time = config.get('min_green_time', self.min_green_time)
            self.max_green_time = config.get('max_green_time', self.max_green_time)
            self.yellow_time = config.get('yellow_time', self.yellow_time)
            self.algorithm = config.get('algorithm', self.algorithm)
            
            # Пересоздаем систему
            if self.algorithm == 'fuzzy' and not self.fuzzy_system:
                self.initialize_fuzzy_system()
            elif self.algorithm == 'webster' and not self.webster_method:
                self.webster_method = WebsterMethod()
            elif self.algorithm == 'reinforcement' and not self.rl_controller:
                self.rl_controller = ReinforcementLearningController()
                
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return False

    def _visualize_fuzzy_system(self):
        """Визуализирует функции принадлежности нечеткой системы"""
        try:
            if 'queue_length' not in self.fuzzy_vars or 'waiting_time' not in self.fuzzy_vars or 'green_time' not in self.fuzzy_vars:
                print("Нечеткие переменные не инициализированы")
                return

            # Создаем фигуру
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            
            # Построение графиков
            self.fuzzy_vars['queue_length'].view(ax=axes[0])
            self.fuzzy_vars['waiting_time'].view(ax=axes[1])
            self.fuzzy_vars['green_time'].view(ax=axes[2])
            
            # Сохраняем и закрываем
            plt.tight_layout()
            plt.savefig('fuzzy_system.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Предупреждение: Не удалось создать визуализацию - {e}")

class WebsterMethod:
    """Метод Вебстера для расчета оптимального времени светофора"""
    def __init__(self, saturation_flow=1800, cycle_length_bounds=(60, 120)):
        self.saturation_flow = saturation_flow  # автомобилей в час
        self.cycle_length_bounds = cycle_length_bounds  # (мин, макс)

    def calculate_optimal_cycle(self, traffic_volumes):
        """Рассчитывает оптимальную длительность цикла по методу Вебстера"""
        # Рассчитываем коэффициенты нагрузки
        ratios = [volume / self.saturation_flow for volume in traffic_volumes]
        total_ratio = sum(ratios)
        
        # Проверяем на перегрузку
        if total_ratio >= 1:
            return self.cycle_length_bounds[1]  # Максимальный цикл
            
        # Формула Вебстера
        optimal_cycle = (1.5 * 4 + 5) / (1 - total_ratio)
        return np.clip(optimal_cycle, *self.cycle_length_bounds)

    def calculate_green_splits(self, traffic_volumes, cycle_length):
        """Рассчитывает распределение зеленого времени между фазами"""
        # Рассчитываем коэффициенты нагрузки
        ratios = [volume / self.saturation_flow for volume in traffic_volumes]
        total_ratio = sum(ratios)
        
        # Если перегрузка, равномерное распределение
        if total_ratio >= 1:
            return [cycle_length/len(traffic_volumes)] * len(traffic_volumes)
            
        # Рассчитываем эффективное зеленое время
        effective_green = cycle_length - 4 * 4  # 4 фазы по 4 секунды
        green_times = [effective_green * ratio / total_ratio for ratio in ratios]
        
        return green_times

class ReinforcementLearningController:
    """Контроллер на основе Q-обучения"""
    def __init__(self, state_space=10, action_space=5, alpha=0.1, 
                 gamma=0.9, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha  # скорость обучения
        self.gamma = gamma  # коэффициент дисконтирования
        self.epsilon = epsilon  # вероятность случайного действия
        
        # Инициализируем Q-таблицу
        self.q_table = np.zeros((state_space, action_space))

    def get_state_index(self, traffic_data):
        """Преобразует данные о трафике в индекс состояния"""
        # Преобразуем данные в число (сумма транспортных средств)
        total_vehicles = sum(traffic_data.values())
        
        # Дискретизируем
        state = int(np.clip(total_vehicles, 0, self.state_space-1))
        return state

    def choose_action(self, state):
        """Выбирает действие согласно стратегии epsilon-greedy"""
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.action_space)  # случайное действие
        else:
            return np.argmax(self.q_table[state, :])  # лучшее известное действие

    def update_q_table(self, state, action, reward, next_state):
        """Обновляет Q-таблицу"""
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        
        # Обновляем Q-значение
        self.q_table[state, action] = (1 - self.alpha) * current_q + \
                                      self.alpha * (reward + self.gamma * max_future_q)

def generate_simulated_traffic():
    """Генерирует имитационные данные о трафике"""
    directions = ['north', 'south', 'east', 'west']
    traffic_data = {}
    
    # Генерируем случайные данные с некоторыми паттернами
    for direction in directions:
        # Основной поток
        base = np.random.poisson(10)
        # Сезонные колебания
        seasonal = 5 * np.sin(2 * np.pi * time.time() / 60)
        # Итоговое значение
        traffic_data[direction] = max(0, int(base + seasonal))
        
    return traffic_data

def main():
    """Основная функция для запуска контроллера"""
    parser = argparse.ArgumentParser(description='Интеллектуальный контроллер светофора')
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

    # Запускаем контроллер
    controller.start()

    # Симуляция
    print(f"\nЗапуск симуляции на {args.simulation_time} секунд с алгоритмом {args.algorithm}...")
    start_time = time.time()

    try:
        while time.time() - start_time < args.simulation_time:
            # Генерируем данные о трафике
            traffic_data = generate_simulated_traffic()
            
            # Обновляем состояние светофора
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