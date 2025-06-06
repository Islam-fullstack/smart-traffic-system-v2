import pygame
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from typing import Dict, List, Tuple, Optional
import os
import argparse
import time

class IntersectionVisualizer:
    """Визуализатор одного перекрестка с анимацией"""
    # Цвета
    COLORS = {
        'background': (255, 255, 255),
        'road': (100, 100, 100),
        'lane': (200, 200, 200),
        'traffic_light': {
            'red': (255, 0, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0)
        },
        'vehicle': {
            'car': (0, 0, 255),
            'truck': (128, 0, 255),
            'bus': (255, 0, 128),
            'motorcycle': (0, 128, 255)
        }
    }

    def __init__(self, window_size: Tuple[int, int], intersection, traffic_light,
                 scale: float = 10.0, fps: int = 30, 
                 background_color: Tuple[int, int, int] = (255, 255, 255)):
        self.window_size = window_size
        self.intersection = intersection
        self.traffic_light = traffic_light
        self.scale = scale
        self.fps = fps
        self.background_color = background_color
        self.screen = None
        self.clock = None
        self.running = False
        self.animation_frames = []

    def initialize_display(self):
        """Инициализирует окно pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Умный светофор - Перекресток")

    def draw_intersection(self):
        """Рисует перекресток с полосами и разметкой"""
        width, height = self.window_size
        center_x, center_y = width // 2, height // 2
        
        # Очистка экрана
        self.screen.fill(self.background_color)
        
        # Рисуем дороги
        road_width = 10 * self.scale  # ширина дороги в пикселях
        for direction in ['north', 'south', 'east', 'west']:
            if direction == 'north':
                rect = pygame.Rect(center_x - road_width//2, 0, road_width, center_y)
            elif direction == 'south':
                rect = pygame.Rect(center_x - road_width//2, center_y, road_width, height - center_y)
            elif direction == 'east':
                rect = pygame.Rect(center_x, center_y - road_width//2, width - center_x, road_width)
            else:  # west
                rect = pygame.Rect(0, center_y - road_width//2, center_x, road_width)
            
            pygame.draw.rect(self.screen, self.COLORS['road'], rect)
            
            # Рисуем полосы
            if direction in ['north', 'south']:
                lane_width = road_width // 2
                for i in [-1, 1]:
                    lane_rect = pygame.Rect(center_x + i * lane_width // 4, 0, 
                                          lane_width // 2, height)
                    pygame.draw.rect(self.screen, self.COLORS['lane'], lane_rect)
            else:
                lane_width = road_width // 2
                for i in [-1, 1]:
                    lane_rect = pygame.Rect(0, center_y + i * lane_width // 4,
                                          width, lane_width // 2)
                    pygame.draw.rect(self.screen, self.COLORS['lane'], lane_rect)

    def draw_vehicles(self, vehicles):
        """Отображает транспортные средства"""
        for vehicle in vehicles:
            x, y = self._calculate_vehicle_position(vehicle)
            size = max(5, vehicle.length * self.scale // 2)
            color = self.COLORS['vehicle'].get(vehicle.vehicle_type, (0, 0, 255))
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
            
            # Индикация направления
            direction = vehicle.direction
            arrow_size = size // 2
            if direction == 'north':
                points = [(x, y), (x - arrow_size, y + arrow_size), (x + arrow_size, y + arrow_size)]
            elif direction == 'south':
                points = [(x, y), (x - arrow_size, y - arrow_size), (x + arrow_size, y - arrow_size)]
            elif direction == 'east':
                points = [(x, y), (x - arrow_size, y - arrow_size), (x - arrow_size, y + arrow_size)]
            else:  # west
                points = [(x, y), (x + arrow_size, y - arrow_size), (x + arrow_size, y + arrow_size)]
            
            pygame.draw.polygon(self.screen, (0, 0, 0), points)

    def _calculate_vehicle_position(self, vehicle):
        """Вычисляет позицию ТС на экране"""
        width, height = self.window_size
        center_x, center_y = width // 2, height // 2
        
        direction = vehicle.direction
        lane_length = 100  # метров
        screen_length = lane_length * self.scale
        
        # Расчет позиции в зависимости от направления
        if direction == 'north':
            x = center_x
            y = center_y - vehicle.position * self.scale
        elif direction == 'south':
            x = center_x
            y = center_y + vehicle.position * self.scale
        elif direction == 'east':
            x = center_x + vehicle.position * self.scale
            y = center_y
        else:  # west
            x = center_x - vehicle.position * self.scale
            y = center_y
            
        return x, y

    def draw_traffic_lights(self):
        """Рисует светофоры для всех направлений"""
        width, height = self.window_size
        center_x, center_y = width // 2, height // 2
        light_radius = 10
        
        # Позиции светофоров
        light_positions = {
            'north': (center_x - 30, center_y - 50),
            'south': (center_x + 30, center_y + 50),
            'east': (center_x + 50, center_y - 30),
            'west': (center_x - 50, center_y + 30)
        }
        
        # Рисуем светофоры
        for direction in light_positions:
            x, y = light_positions[direction]
            pygame.draw.rect(self.screen, (128, 128, 128), (x - 10, y - 40, 20, 80))
            
            # Получаем текущее состояние
            state = self.traffic_light.get_state().get(direction, 'red')
            color = self.COLORS['traffic_light'].get(state, (128, 128, 128))
            
            # Рисуем сигнал
            if direction in ['north', 'south']:
                pygame.draw.circle(self.screen, color, (x, y - 20), light_radius)
                pygame.draw.circle(self.screen, (128, 128, 128), (x, y), light_radius)
                pygame.draw.circle(self.screen, (128, 128, 128), (x, y + 20), light_radius)
            else:
                pygame.draw.circle(self.screen, (128, 128, 128), (x - 20, y), light_radius)
                pygame.draw.circle(self.screen, color, (x, y), light_radius)
                pygame.draw.circle(self.screen, (128, 128, 128), (x + 20, y), light_radius)

    def draw_metrics(self, metrics):
        """Отображает метрики эффективности"""
        font = pygame.font.SysFont(None, 24)
        
        # Время ожидания
        wait_time = metrics.get('average_waiting_time', 0)
        text = font.render(f"Среднее время ожидания: {wait_time:.1f}с", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        # Пропускная способность
        throughput = metrics.get('throughput', 0)
        text = font.render(f"Пропускная способность: {throughput} ТС/мин", True, (0, 0, 0))
        self.screen.blit(text, (10, 40))
        
        # Активная фаза
        phase = metrics.get('active_phase', '')
        text = font.render(f"Активная фаза: {phase}", True, (0, 0, 0))
        self.screen.blit(text, (10, 70))

    def update(self, simulation_state):
        """Обновляет визуализацию на основе текущего состояния симуляции"""
        self.draw_intersection()
        self.draw_traffic_lights()
        
        # Получаем транспортные средства
        vehicles = simulation_state.get('vehicles', [])
        self.draw_vehicles(vehicles)
        
        # Отображаем метрики
        self.draw_metrics(simulation_state)
        
        # Сохраняем кадр для анимации
        if self.running:
            frame = pygame.surfarray.array3d(self.screen)
            self.animation_frames.append(np.transpose(frame, (1, 0, 2)))
            
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run_animation(self, simulation):
        """Запускает анимацию в реальном времени"""
        self.running = True
        self.initialize_display()
        
        try:
            for frame in simulation:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                
                if not self.running:
                    break
                
                self.update(frame)
        finally:
            pygame.quit()

    def save_animation(self, output_file, simulation):
        """Сохраняет анимацию в файл"""
        self.running = False
        self.initialize_display()
        
        temp_frames = []
        try:
            for frame in simulation:
                self.update(frame)
                frame = pygame.surfarray.array3d(self.screen)
                temp_frames.append(np.transpose(frame, (1, 0, 2)))
                
            clip = ImageSequenceClip(temp_frames, fps=self.fps)
            clip.write_videofile(output_file, codec='libx264')
        finally:
            pygame.quit()

class ComparisonVisualizer:
    """Визуализатор для сравнения контроллеров"""
    def __init__(self, window_size: Tuple[int, int], traditional_intersection, 
                 smart_intersection, scale: float = 10.0, fps: int = 30):
        self.window_size = window_size
        self.traditional_intersection = traditional_intersection
        self.smart_intersection = smart_intersection
        self.scale = scale
        self.fps = fps
        self.screen = None
        self.clock = None
        self.running = False
        self.animation_frames = []

    def initialize_display(self):
        """Инициализирует окно pygame"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Сравнение контроллеров светофора")

    def draw_split_view(self):
        """Рисует оба перекрестка рядом"""
        width, height = self.window_size
        half_width = width // 2
        
        # Очистка экрана
        self.screen.fill((255, 255, 255))
        
        # Рисуем разделительную линию
        pygame.draw.line(self.screen, (0, 0, 0), (half_width, 0), (half_width, height), 2)
        
        # Рисуем левые и правые перекрестки
        self._draw_single_intersection(self.traditional_intersection, (0, 0))
        self._draw_single_intersection(self.smart_intersection, (half_width, 0))

    def _draw_single_intersection(self, intersection, offset):
        """Рисует один перекресток с заданным смещением"""
        width, height = self.window_size
        center_x, center_y = width // 4, height // 2
        center_x += offset[0]
        center_y += offset[1]
        
        road_width = 10 * self.scale
        # Рисуем дороги
        for direction in ['north', 'south', 'east', 'west']:
            if direction == 'north':
                rect = pygame.Rect(center_x - road_width//2, 0 + offset[1], 
                                  road_width, center_y)
            elif direction == 'south':
                rect = pygame.Rect(center_x - road_width//2, center_y + offset[1], 
                                  road_width, height - center_y)
            elif direction == 'east':
                rect = pygame.Rect(center_x + offset[0], center_y - road_width//2 + offset[1], 
                                  width - center_x, road_width)
            else:  # west
                rect = pygame.Rect(0 + offset[0], center_y - road_width//2 + offset[1], 
                                  center_x, road_width)
            
            pygame.draw.rect(self.screen, (100, 100, 100), rect)

    def update(self, traditional_state, smart_state):
        """Обновляет визуализацию обоих перекрестков"""
        self.draw_split_view()
        
        # Получаем и отрисовываем ТС для традиционного контроллера
        traditional_vehicles = traditional_state.get('vehicles', [])
        self._draw_vehicles(traditional_vehicles, (0, 0))
        
        # Получаем и отрисовываем ТС для умного контроллера
        smart_vehicles = smart_state.get('vehicles', [])
        half_width = self.window_size[0] // 2
        self._draw_vehicles(smart_vehicles, (half_width, 0))
        
        # Рисуем светофоры
        self._draw_traffic_lights(
            self.traditional_intersection.traffic_light.get_state(), 
            (0, 0)
        )
        self._draw_traffic_lights(
            self.smart_intersection.traffic_light.get_state(), 
            (half_width, 0)
        )
        
        # Отображаем метрики
        self._draw_metrics(traditional_state, (0, 0))
        self._draw_metrics(smart_state, (half_width, 0))
        
        # Сохраняем кадр
        if self.running:
            frame = pygame.surfarray.array3d(self.screen)
            self.animation_frames.append(np.transpose(frame, (1, 0, 2)))
            
        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_vehicles(self, vehicles, offset):
        """Отображает ТС с заданным смещением"""
        for vehicle in vehicles:
            x, y = self._calculate_vehicle_position(vehicle, offset)
            size = max(5, vehicle.length * self.scale // 2)
            color = (0, 0, 255)  # Дефолтный цвет
            
            # Получаем цвет в зависимости от типа
            if hasattr(vehicle, 'vehicle_type'):
                color = IntersectionVisualizer.COLORS['vehicle'].get(vehicle.vehicle_type, color)
                
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

    def _calculate_vehicle_position(self, vehicle, offset):
        """Вычисляет позицию ТС на экране с учетом смещения"""
        width, height = self.window_size
        center_x, center_y = width // 4, height // 2
        center_x += offset[0]
        center_y += offset[1]
        
        direction = vehicle.direction
        lane_length = 100  # метров
        screen_length = lane_length * self.scale
        
        # Расчет позиции
        if direction == 'north':
            x = center_x
            y = center_y - vehicle.position * self.scale
        elif direction == 'south':
            x = center_x
            y = center_y + vehicle.position * self.scale
        elif direction == 'east':
            x = center_x + vehicle.position * self.scale
            y = center_y
        else:  # west
            x = center_x - vehicle.position * self.scale
            y = center_y
            
        return x, y

    def _draw_traffic_lights(self, light_state, offset):
        """Рисует светофоры с заданным смещением"""
        width, height = self.window_size
        center_x, center_y = width // 4, height // 2
        center_x += offset[0]
        center_y += offset[1]
        
        # Позиции светофоров
        light_positions = {
            'north': (center_x - 30, center_y - 50),
            'south': (center_x + 30, center_y + 50),
            'east': (center_x + 50, center_y - 30),
            'west': (center_x - 50, center_y + 30)
        }
        
        # Рисуем светофоры
        for direction, pos in light_positions.items():
            x, y = pos
            pygame.draw.rect(self.screen, (128, 128, 128), (x - 10, y - 40, 20, 80))
            
            # Получаем текущее состояние
            state = light_state.get(direction, 'red')
            color = IntersectionVisualizer.COLORS['traffic_light'].get(state, (128, 128, 128))
            
            # Рисуем сигнал
            if direction in ['north', 'south']:
                pygame.draw.circle(self.screen, color, (x, y - 20), 10)
            else:
                pygame.draw.circle(self.screen, color, (x, y), 10)

    def _draw_metrics(self, metrics, offset):
        """Отображает метрики с заданным смещением"""
        font = pygame.font.SysFont(None, 24)
        
        # Время ожидания
        wait_time = metrics.get('average_waiting_time', 0)
        text = font.render(f"Ср. время ожидания: {wait_time:.1f}с", True, (0, 0, 0))
        self.screen.blit(text, (10 + offset[0], 10 + offset[1]))

    def run_comparison_animation(self, traditional_sim, smart_sim):
        """Запускает сравнительную анимацию"""
        self.running = True
        self.initialize_display()
        
        try:
            for trad_frame, smart_frame in zip(traditional_sim, smart_sim):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                
                if not self.running:
                    break
                
                self.update(trad_frame, smart_frame)
        finally:
            pygame.quit()

    def save_comparison_animation(self, output_file, traditional_sim, smart_sim):
        """Сохраняет сравнительную анимацию"""
        self.running = False
        self.initialize_display()
        
        try:
            for trad_frame, smart_frame in zip(traditional_sim, smart_sim):
                self.update(trad_frame, smart_frame)
        finally:
            pygame.quit()
            
        clip = ImageSequenceClip(self.animation_frames, fps=self.fps)
        clip.write_videofile(output_file, codec='libx264')

class MetricsPlotter:
    """Класс для построения графиков метрик"""
    def __init__(self, traditional_data: List[Dict], smart_data: List[Dict], output_dir: str):
        self.traditional_data = traditional_data
        self.smart_data = smart_data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_waiting_time(self):
        """Создает график среднего времени ожидания"""
        trad_times = [frame['average_waiting_time'] for frame in self.traditional_data]
        smart_times = [frame['average_waiting_time'] for frame in self.smart_data]
        times = list(range(len(trad_times)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, trad_times, label='Традиционный контроллер')
        plt.plot(times, smart_times, label='Умный контроллер')
        
        plt.xlabel('Время (кадры)')
        plt.ylabel('Среднее время ожидания (секунды)')
        plt.title('Сравнение среднего времени ожидания')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'waiting_time_comparison.png'))
        plt.close()

    def plot_queue_length(self):
        """Создает график длины очередей"""
        trad_queues = [max(frame['queue_lengths'].values()) for frame in self.traditional_data]
        smart_queues = [max(frame['queue_lengths'].values()) for frame in self.smart_data]
        times = list(range(len(trad_queues)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, trad_queues, label='Традиционный контроллер')
        plt.plot(times, smart_queues, label='Умный контроллер')
        
        plt.xlabel('Время (кадры)')
        plt.ylabel('Максимальная длина очереди')
        plt.title('Сравнение длины очередей')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'queue_length_comparison.png'))
        plt.close()

    def plot_throughput(self):
        """Создает график пропускной способности"""
        trad_throughput = [frame['throughput'] for frame in self.traditional_data]
        smart_throughput = [frame['throughput'] for frame in self.smart_data]
        times = list(range(len(trad_throughput)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, trad_throughput, label='Традиционный контроллер')
        plt.plot(times, smart_throughput, label='Умный контроллер')
        
        plt.xlabel('Время (кадры)')
        plt.ylabel('Пропускная способность')
        plt.title('Сравнение пропускной способности')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'throughput_comparison.png'))
        plt.close()

    def plot_fuel_consumption(self):
        """Создает график расхода топлива"""
        trad_fuel = [frame.get('fuel_consumption', 0) for frame in self.traditional_data]
        smart_fuel = [frame.get('fuel_consumption', 0) for frame in self.smart_data]
        times = list(range(len(trad_fuel)))
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, trad_fuel, label='Традиционный контроллер')
        plt.plot(times, smart_fuel, label='Умный контроллер')
        
        plt.xlabel('Время (кадры)')
        plt.ylabel('Расход топлива')
        plt.title('Сравнение расхода топлива')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.output_dir, 'fuel_consumption_comparison.png'))
        plt.close()

    def create_summary_plot(self):
        """Создает общий отчет с графиками"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Время ожидания
        trad_times = [frame['average_waiting_time'] for frame in self.traditional_data]
        smart_times = [frame['average_waiting_time'] for frame in self.smart_data]
        axs[0, 0].plot(trad_times, label='Традиционный')
        axs[0, 0].plot(smart_times, label='Умный')
        axs[0, 0].set_title('Среднее время ожидания')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Длина очередей
        trad_queues = [max(frame['queue_lengths'].values()) for frame in self.traditional_data]
        smart_queues = [max(frame['queue_lengths'].values()) for frame in self.smart_data]
        axs[0, 1].plot(trad_queues, label='Традиционный')
        axs[0, 1].plot(smart_queues, label='Умный')
        axs[0, 1].set_title('Максимальная длина очереди')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Пропускная способность
        trad_throughput = [frame['throughput'] for frame in self.traditional_data]
        smart_throughput = [frame['throughput'] for frame in self.smart_data]
        axs[1, 0].plot(trad_throughput, label='Традиционный')
        axs[1, 0].plot(smart_throughput, label='Умный')
        axs[1, 0].set_title('Пропускная способность')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Расход топлива
        trad_fuel = [frame.get('fuel_consumption', 0) for frame in self.traditional_data]
        smart_fuel = [frame.get('fuel_consumption', 0) for frame in self.smart_data]
        axs[1, 1].plot(trad_fuel, label='Традиционный')
        axs[1, 1].plot(smart_fuel, label='Умный')
        axs[1, 1].set_title('Расход топлива')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'summary_report.png'))
        plt.close()

    def save_all_plots(self):
        """Сохраняет все графики"""
        self.plot_waiting_time()
        self.plot_queue_length()
        self.plot_throughput()
        self.plot_fuel_consumption()
        self.create_summary_plot()

class DashboardUI:
    """Интерактивный интерфейс управления симуляцией"""
    def __init__(self, simulation_manager, visualizer):
        self.simulation_manager = simulation_manager
        self.visualizer = visualizer
        self.screen = None
        self.clock = None
        self.running = False
        self.mode = 'traditional'
        self.speed = 1.0
        self.font = None

    def create_control_panel(self):
        """Создает панель управления симуляцией"""
        panel_width = 200
        panel_height = self.window_size[1]
        control_panel = pygame.Surface((panel_width, panel_height))
        control_panel.fill((230, 230, 230))
        
        # Заголовок
        title_font = pygame.font.SysFont(None, 24)
        title = title_font.render("Панель управления", True, (0, 0, 0))
        control_panel.blit(title, (10, 10))
        
        # Кнопки
        button_font = pygame.font.SysFont(None, 20)
        
        # Кнопка запуска
        self.start_button = pygame.Rect(10, 50, 180, 40)
        pygame.draw.rect(control_panel, (0, 200, 0), self.start_button)
        start_text = button_font.render("Запуск", True, (255, 255, 255))
        control_panel.blit(start_text, (70, 60))
        
        # Кнопка паузы
        self.pause_button = pygame.Rect(10, 100, 180, 40)
        pygame.draw.rect(control_panel, (200, 200, 0), self.pause_button)
        pause_text = button_font.render("Пауза", True, (255, 255, 255))
        control_panel.blit(pause_text, (70, 110))
        
        # Кнопка остановки
        self.stop_button = pygame.Rect(10, 150, 180, 40)
        pygame.draw.rect(control_panel, (200, 0, 0), self.stop_button)
        stop_text = button_font.render("Остановка", True, (255, 255, 255))
        control_panel.blit(stop_text, (60, 160))
        
        # Переключатель контроллеров
        switch_font = pygame.font.SysFont(None, 20)
        switch_label = switch_font.render("Контроллер:", True, (0, 0, 0))
        control_panel.blit(switch_label, (10, 210))
        
        self.traditional_button = pygame.Rect(10, 230, 180, 30)
        self.smart_button = pygame.Rect(10, 265, 180, 30)
        
        pygame.draw.rect(control_panel, (100, 100, 255), self.traditional_button)
        pygame.draw.rect(control_panel, (100, 255, 100), self.smart_button)
        
        traditional_text = switch_font.render("Традиционный", True, (255, 255, 255))
        smart_text = switch_font.render("Умный", True, (255, 255, 255))
        control_panel.blit(traditional_text, (35, 235))
        control_panel.blit(smart_text, (60, 270))
        
        return control_panel

    def create_metrics_panel(self, metrics):
        """Создает панель отображения метрик"""
        metrics_panel = pygame.Surface((200, 150))
        metrics_panel.fill((245, 245, 245))
        
        font = pygame.font.SysFont(None, 20)
        
        # Заголовок
        title = font.render("Текущие метрики", True, (0, 0, 0))
        metrics_panel.blit(title, (10, 10))
        
        # Отображаем метрики
        y_pos = 40
        for metric, value in metrics.items():
            if metric in ['average_waiting_time', 'max_queue_length', 'throughput', 'fuel_consumption']:
                label = font.render(f"{metric}: {value:.2f}", True, (0, 0, 0))
                metrics_panel.blit(label, (10, y_pos))
                y_pos += 30
        
        return metrics_panel

    def handle_events(self):
        """Обрабатывает события пользовательского интерфейса"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Обработка кликов по панели управления
                mouse_pos = pygame.mouse.get_pos()
                
                # Кнопка запуска
                if self.start_button.collidepoint(mouse_pos):
                    print("Запуск симуляции")
                    
                # Кнопка паузы
                elif self.pause_button.collidepoint(mouse_pos):
                    self.simulation_manager.toggle_pause()
                    
                # Кнопка остановки
                elif self.stop_button.collidepoint(mouse_pos):
                    self.running = False
                    
                # Переключатель контроллеров
                elif self.traditional_button.collidepoint(mouse_pos):
                    self.mode = 'traditional'
                elif self.smart_button.collidepoint(mouse_pos):
                    self.mode = 'smart'

    def run(self):
        """Запускает интерактивный режим"""
        self.running = True
        self.window_size = (self.visualizer.window_size[0] + 200, self.visualizer.window_size[1])
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        try:
            while self.running:
                self.handle_events()
                
                # Получаем текущее состояние
                if self.mode == 'traditional':
                    simulation_state = self.simulation_manager.run_traditional_simulation()
                else:
                    simulation_state = self.simulation_manager.run_smart_simulation()
                
                # Рисуем основную сцену
                self.visualizer.update(simulation_state)
                
                # Рисуем панель управления
                control_panel = self.create_control_panel()
                self.screen.blit(control_panel, (self.visualizer.window_size[0], 0))
                
                # Рисуем метрики
                metrics_panel = self.create_metrics_panel(simulation_state)
                self.screen.blit(metrics_panel, (self.visualizer.window_size[0], 300))
                
                pygame.display.flip()
                self.clock.tick(self.visualizer.fps)
        finally:
            pygame.quit()

def main():
    """Основная функция для запуска визуализации"""
    parser = argparse.ArgumentParser(description='Визуализация системы умного светофора')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'comparison'],
                        help='Режим визуализации')
    parser.add_argument('--window-size', type=str, default='800,600',
                        help='Размер окна (ширина,высота)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Частота кадров')
    parser.add_argument('--output-dir', type=str, default='visualization_results',
                        help='Директория для сохранения результатов')
    parser.add_argument('--save-animation', action='store_true',
                        help='Сохранить анимацию')
    parser.add_argument('--scale', type=float, default=10,
                        help='Масштаб визуализации')
    parser.add_argument('--interactive', action='store_true',
                        help='Интерактивный режим')
    args = parser.parse_args()

    # Создаем временные заглушки для Intersection и TrafficLight
    class DummyIntersection:
        pass
    
    class DummyTrafficLight:
        def get_state(self):
            return {'north': 'green', 'south': 'red', 'east': 'red', 'west': 'red'}

    # Инициализируем визуализаторы
    window_size = tuple(map(int, args.window_size.split(',')))
    
    intersection = DummyIntersection()
    traffic_light = DummyTrafficLight()
    
    visualizer = IntersectionVisualizer(
        window_size=window_size,
        intersection=intersection,
        traffic_light=traffic_light,
        scale=args.scale,
        fps=args.fps
    )
    
    # Запускаем анимацию
    if args.interactive:
        dashboard = DashboardUI(None, visualizer)
        dashboard.run()
    else:
        # Генерируем тестовые данные
        test_data = [{
            'average_waiting_time': np.random.uniform(0, 60),
            'throughput': np.random.randint(0, 20),
            'queue_lengths': {d: np.random.randint(0, 10) for d in ['north', 'south', 'east', 'west']},
            'active_phase': 'North-South Green',
            'fuel_consumption': np.random.uniform(0, 5)
        } for _ in range(100)]
        
        if args.save_animation:
            output_path = os.path.join(args.output_dir, 'simulation.mp4')
            visualizer.save_animation(output_path, test_data)
        else:
            visualizer.run_animation(test_data)

if __name__ == "__main__":
    main()