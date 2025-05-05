import numpy as np
import argparse
from collections import deque

class Vehicle:
    """Класс для представления транспортного средства"""
    DEFAULT_PROPERTIES = {
        'car': {'speed': 20, 'length': 5},  # Скорость в м/с, длина в метрах
        'truck': {'speed': 15, 'length': 10},
        'bus': {'speed': 18, 'length': 12},
        'motorcycle': {'speed': 25, 'length': 3}
    }

    def __init__(self, arrival_time, vehicle_type, direction):
        self.arrival_time = arrival_time
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.position = 100  # начальное расстояние до перекрестка
        self.length = {
            'car': 5,
            'truck': 10,
            'bus': 12,
            'motorcycle': 3
        }.get(vehicle_type, 5)  # по умолчанию 5 метров

    def update_position(self, time_step, is_moving=True):
        """Обновляет позицию ТС"""
        if is_moving:
            self.position -= 10 * time_step  # 10 м/с - примерная скорость
        else:
            if self.position > 0:
                self.position = max(0, self.position)

    def calculate_waiting_time(self, current_time):
        """Возвращает время ожидания на светофоре"""
        return max(0, current_time - self.arrival_time) if self.position > 0 else 0

class Lane:
    """Класс для представления полосы движения"""
    def __init__(self, lane_id, direction, length, max_speed):
        self.lane_id = lane_id
        self.direction = direction
        self.length = length
        self.max_speed = max_speed
        self.vehicles = deque()  # Очередь транспортных средств

    def add_vehicle(self, vehicle):
        """Добавляет транспортное средство в полосу"""
        vehicle.position = self.length  # Начальная позиция - длина полосы
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle_id):
        """Удаляет транспортное средство по ID"""
        for i, v in enumerate(self.vehicles):
            if v.vehicle_id == vehicle_id:
                del self.vehicles[i]
                break

    def get_queue_length(self):
        """Возвращает длину очереди в полосе"""
        return len(self.vehicles)

    def update(self, green_light, time_step):
        """Обновляет состояние всех транспортных средств в полосе"""
        passed_vehicles = 0
        new_vehicles = []
        
        for v in self.vehicles:
            v.update_position(green_light, time_step)
            if v.position > 0:
                new_vehicles.append(v)
            else:
                passed_vehicles += 1
                
        self.vehicles = deque(new_vehicles)
        return passed_vehicles

class TrafficLight:
    """Класс для управления светофором"""
    def __init__(self, phases=None, phase_durations=None):
        if phases is None:
            self.phases = ['north', 'south', 'east', 'west']
            self.phase_durations = [30, 30, 30, 30]  # Длительность фазы в секундах
        else:
            self.phases = phases
            self.phase_durations = phase_durations
            
        self.current_phase_index = 0
        self.current_phase = self.phases[self.current_phase_index]
        self.phase_start_time = 0

    def update(self, current_time):
        """Обновляет состояние светофора"""
        if current_time - self.phase_start_time >= self.phase_durations[self.current_phase_index]:
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)
            self.current_phase = self.phases[self.current_phase_index]
            self.phase_start_time = current_time
        return self.current_phase

class Intersection:
    """Класс для представления перекрестка"""
    def __init__(self, lanes, traffic_light=None):
        self.lanes = {lane.direction: lane for lane in lanes}
        self.traffic_light = traffic_light
        self.passed_vehicles_count = 0
        self.current_time = 0

    def update(self, current_time, time_step):
        """Обновляет состояние перекрестка"""
        self.current_time = current_time
        self.traffic_light.update(current_time)
        green_direction = self.traffic_light.current_phase

        # Обновляем каждую полосу
        for direction, lane in self.lanes.items():
            is_green = (direction == green_direction)
            passed = lane.update(is_green, time_step)
            self.passed_vehicles_count += passed

    def add_vehicle(self, vehicle):
        """Добавляет транспортное средство в соответствующую полосу"""
        if vehicle.direction in self.lanes:
            self.lanes[vehicle.direction].add_vehicle(vehicle)
        else:
            raise ValueError(f"Направление {vehicle.direction} не найдено в перекрестке")

    def get_metrics(self):
        """Возвращает метрики перекрестка"""
        queue_lengths = {direction: lane.get_queue_length() for direction, lane in self.lanes.items()}
        
        total_waiting_time = 0
        total_vehicles = 0
        
        for lane in self.lanes.values():
            for vehicle in lane.vehicles:
                waiting_time = vehicle.calculate_waiting_time(self.current_time)
                total_waiting_time += waiting_time
                total_vehicles += 1
                
        average_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0

        return {
            'queue_lengths': queue_lengths,
            'average_waiting_time': average_waiting_time,
            'throughput': self.passed_vehicles_count
        }

class TrafficSimulation:
    """Класс для моделирования транспортного потока"""
    def __init__(self, intersection, arrival_rates, vehicle_type_distribution, simulation_time, time_step=1.0):
        self.intersection = intersection
        self.arrival_rates = arrival_rates
        self.vehicle_type_distribution = vehicle_type_distribution
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.current_time = 0
        self.vehicle_counter = 0
        self.metrics_history = []

    def generate_vehicles(self):
        """Генерирует новые транспортные средства"""
        for direction, rate in self.arrival_rates.items():
            if np.random.uniform() < self._calculate_probability(rate):
                vehicle_type = self._select_vehicle_type()
                vehicle = Vehicle(
                    arrival_time=self.current_time,
                    vehicle_type=vehicle_type,
                    direction=direction
                )
                if direction in self.intersection.lanes:
                    self.intersection.lanes[direction].add_vehicle(vehicle)

    def run_simulation(self):
        """Запускает симуляцию"""
        self.current_time = 0
        self.metrics_history = []
        
        while self.current_time < self.simulation_time:
            self.generate_vehicles(self.current_time)
            self.intersection.update(self.current_time, self.time_step)
            metrics = self.intersection.get_metrics()
            
            self.metrics_history.append({
                'time': self.current_time,
                'metrics': metrics
            })
            
            self.current_time += self.time_step
            
        return self.metrics_history

    def get_results(self):
        """Возвращает результаты симуляции"""
        if not self.metrics_history:
            return {
                'total_passed': 0,
                'average_waiting_time': 0,
                'total_time': 0
            }
            
        total_passed = self.intersection.passed_vehicles_count
        total_waiting_time = sum(m['metrics']['average_waiting_time'] for m in self.metrics_history)
        average_waiting_time = total_waiting_time / len(self.metrics_history)
        
        return {
            'total_passed': total_passed,
            'average_waiting_time': average_waiting_time,
            'total_time': self.simulation_time
        }

def main():
    """Основная функция для запуска симуляции"""
    parser = argparse.ArgumentParser(description='Моделирование транспортного потока на перекрестке')
    parser.add_argument('--simulation-time', type=int, default=3600, help='Время симуляции в секундах')
    parser.add_argument('--north-rate', type=int, default=10, help='Интенсивность с севера (ТС/мин)')
    parser.add_argument('--south-rate', type=int, default=10, help='Интенсивность с юга (ТС/мин)')
    parser.add_argument('--east-rate', type=int, default=15, help='Интенсивность с востока (ТС/мин)')
    parser.add_argument('--west-rate', type=int, default=15, help='Интенсивность с запада (ТС/мин)')
    args = parser.parse_args()

    # Создаем полосы движения
    lanes = [
        Lane(lane_id=0, direction='north', length=100, max_speed=20),
        Lane(lane_id=1, direction='south', length=100, max_speed=20),
        Lane(lane_id=2, direction='east', length=100, max_speed=20),
        Lane(lane_id=3, direction='west', length=100, max_speed=20)
    ]
    
    # Создаем светофор
    traffic_light = TrafficLight()
    
    # Создаем перекресток
    intersection = Intersection(lanes=lanes, traffic_light=traffic_light)
    
    # Параметры интенсивности
    arrival_rates = {
        'north': args.north_rate,
        'south': args.south_rate,
        'east': args.east_rate,
        'west': args.west_rate
    }
    
    # Распределение типов транспорта
    vehicle_type_distribution = {
        'car': 0.6,
        'truck': 0.1,
        'bus': 0.2,
        'motorcycle': 0.1
    }
    
    # Создаем симуляцию
    simulation = TrafficSimulation(
        intersection=intersection,
        arrival_rates=arrival_rates,
        vehicle_type_distribution=vehicle_type_distribution,
        simulation_time=args.simulation_time
    )
    
    # Запускаем симуляцию
    print("Запуск симуляции...")
    simulation.run_simulation()
    
    # Получаем и выводим результаты
    results = simulation.get_results()
    print("\nРезультаты симуляции:")
    print(f"Общее время симуляции: {results['total_time']} секунд")
    print(f"Общее количество прошедших ТС: {results['total_passed']}")
    print(f"Среднее время ожидания: {results['average_waiting_time']:.2f} секунд")

if __name__ == "__main__":
    main()