import time
import json
import argparse
from enum import Enum

class TrafficLightColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"

class TrafficLightPhase:
    """Класс для управления фазой светофора"""
    def __init__(self, phase_id, phase_name, durations, directions):
        self.phase_id = phase_id
        self.phase_name = phase_name
        self.durations = durations
        self.directions = directions
        self.total_duration = self.get_total_duration()

    def get_total_duration(self):
        """Возвращает общую продолжительность фазы"""
        return sum(self.durations.values())

    def get_state_at_time(self, time_in_phase):
        """Возвращает состояние всех направлений в заданный момент времени внутри фазы"""
        current_time = 0
        
        # Проверяем зеленый сигнал
        if current_time <= time_in_phase < current_time + self.durations["green"]:
            return {dir: "green" for dir, state in self.directions.items()}
        
        current_time += self.durations["green"]
        
        # Проверяем желтый сигнал
        if current_time <= time_in_phase < current_time + self.durations["yellow"]:
            return {dir: "yellow" for dir, state in self.directions.items()}
        
        current_time += self.durations["yellow"]
        
        # Проверяем красный сигнал
        if current_time <= time_in_phase < current_time + self.durations["red"]:
            return {dir: "red" for dir, state in self.directions.items()}
        
        # По умолчанию возвращаем текущее состояние
        return self.directions

class TraditionalController:
    """Класс для управления светофором с фиксированными фазами"""
    def __init__(self, phases, cycle_times=None, current_mode="day"):
        self.phases = phases
        self.cycle_times = cycle_times or {
            "morning": 120,
            "day": 90,
            "evening": 120,
            "night": 60
        }
        self.current_mode = current_mode
        self.start_time = None
        self.total_cycle_duration = sum(phase.total_duration for phase in self.phases)

    def start(self):
        """Запускает работу контроллера и инициализирует таймер"""
        self.start_time = time.time()

    def update(self, current_time):
        """Обновляет состояние светофора"""
        if self.start_time is None:
            self.start()
        
        # Рассчитываем время с момента запуска
        elapsed_time = current_time - self.start_time if isinstance(current_time, float) else time.time() - self.start_time
        
        # Получаем общий цикл
        cycle_duration = self.cycle_times[self.current_mode]
        
        # Нормализуем время в рамках цикла
        normalized_time = elapsed_time % cycle_duration
        
        # Определяем текущую фазу
        time_counter = 0
        for phase in self.phases:
            if normalized_time < time_counter + phase.total_duration:
                time_in_phase = normalized_time - time_counter
                return phase.get_state_at_time(time_in_phase)
            time_counter += phase.total_duration
            
        # По умолчанию возвращаем последнюю фазу
        return self.phases[-1].get_state_at_time(self.phases[-1].total_duration - 1)

    def set_mode(self, mode):
        """Изменяет режим работы"""
        if mode in self.cycle_times:
            self.current_mode = mode
        else:
            raise ValueError(f"Режим {mode} не найден в настройках")

    def get_current_phase(self):
        """Возвращает текущую активную фазу"""
        current_time = time.time()
        if self.start_time is None:
            self.start()
            
        elapsed_time = current_time - self.start_time
        normalized_time = elapsed_time % self.total_cycle_duration
        
        time_counter = 0
        for phase in self.phases:
            if normalized_time < time_counter + phase.total_duration:
                return phase
            time_counter += phase.total_duration
            
        return self.phases[-1]

    def load_configuration(self, config_file):
        """Загружает конфигурацию из JSON файла"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Загружаем фазы
            phases = []
            for phase_data in config.get('phases', []):
                phase = TrafficLightPhase(
                    phase_id=phase_data['phase_id'],
                    phase_name=phase_data['phase_name'],
                    durations=phase_data['durations'],
                    directions=phase_data['directions']
                )
                phases.append(phase)
            
            self.phases = phases
            
            # Обновляем общую длительность цикла
            self.total_cycle_duration = sum(phase.total_duration for phase in self.phases)
            
            # Загружаем режимы работы
            if 'cycle_times' in config:
                self.cycle_times = config['cycle_times']
                
            if 'default_mode' in config:
                self.current_mode = config['default_mode']
                
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            return False

    def save_state(self, state_file):
        """Сохраняет текущее состояние в файл"""
        try:
            state = {
                "phases": [{
                    "phase_id": phase.phase_id,
                    "phase_name": phase.phase_name,
                    "durations": phase.durations,
                    "directions": phase.directions
                } for phase in self.phases],
                "cycle_times": self.cycle_times,
                "current_mode": self.current_mode,
                "start_time": self.start_time
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения состояния: {e}")
            return False

def create_default_phases():
    """Создает стандартный набор фаз для четырехстороннего перекрестка"""
    phases = [
        TrafficLightPhase(
            phase_id=0,
            phase_name="North-South Green",
            durations={"green": 30, "yellow": 5, "red": 0},
            directions={
                "north": "green", 
                "south": "green", 
                "east": "red", 
                "west": "red"
            }
        ),
        TrafficLightPhase(
            phase_id=1,
            phase_name="North-South Yellow",
            durations={"green": 0, "yellow": 5, "red": 0},
            directions={
                "north": "yellow", 
                "south": "yellow", 
                "east": "red", 
                "west": "red"
            }
        ),
        TrafficLightPhase(
            phase_id=2,
            phase_name="East-West Green",
            durations={"green": 30, "yellow": 5, "red": 0},
            directions={
                "north": "red", 
                "south": "red", 
                "east": "green", 
                "west": "green"
            }
        ),
        TrafficLightPhase(
            phase_id=3,
            phase_name="East-West Yellow",
            durations={"green": 0, "yellow": 5, "red": 0},
            directions={
                "north": "red", 
                "south": "red", 
                "east": "yellow", 
                "west": "yellow"
            }
        )
    ]
    return phases

class TrafficLight:
    """Класс для управления светофором"""
    def __init__(self, controller, directions=["north", "south", "east", "west"]):
        self.controller = controller
        self.directions = directions
        self.current_state = {dir: "red" for dir in directions}

    def update(self, current_time=None):
        """Обновляет состояние светофора"""
        self.current_state = self.controller.update(current_time or time.time())

    def get_state(self):
        """Возвращает текущее состояние всех направлений"""
        return self.current_state

    def is_green(self, direction):
        """Проверяет, горит ли зеленый свет для указанного направления"""
        return self.current_state.get(direction) == "green"

    def get_active_directions(self):
        """Возвращает направления с зеленым светом"""
        return [dir for dir, state in self.current_state.items() if state == "green"]

def main():
    """Основная функция для запуска контроллера"""
    parser = argparse.ArgumentParser(description='Классический светофорный контроллер')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='day', choices=['morning', 'day', 'evening', 'night'],
                        help='Режим работы (утро, день, вечер, ночь)')
    parser.add_argument('--simulation-time', type=int, default=300,
                        help='Время симуляции в секундах')
    parser.add_argument('--output', type=str, help='Путь для сохранения состояний')
    args = parser.parse_args()

    # Создаем стандартные фазы
    if args.config:
        print(f"Загрузка конфигурации из {args.config}...")
        phases = []  # Будет заполнено из конфигурации
        controller = TraditionalController(phases)
        if not controller.load_configuration(args.config):
            print("Используем стандартные фазы вместо загруженных")
            phases = create_default_phases()
            controller = TraditionalController(phases)
    else:
        print("Используем стандартные фазы...")
        phases = create_default_phases()
        controller = TraditionalController(phases)

    # Устанавливаем режим работы
    controller.set_mode(args.mode)
    
    # Создаем светофор
    traffic_light = TrafficLight(controller)
    
    # Сохраняем начальное состояние
    if args.output:
        controller.save_state(args.output)
    
    # Запускаем симуляцию
    print(f"\nЗапуск симуляции на {args.simulation_time} секунд в режиме '{args.mode}'...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.simulation_time:
            traffic_light.update()
            current_state = traffic_light.get_state()
            active_dirs = traffic_light.get_active_directions()
            
            print(f"\nВремя: {int(time.time() - start_time)} сек")
            print(f"Текущее состояние: {current_state}")
            print(f"Активные направления: {active_dirs}")
            
            # Ждем 5 секунд перед следующим обновлением
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nСимуляция остановлена пользователем")
    
    print("\nСимуляция завершена")

if __name__ == "__main__":
    main()