### ✅ README.md

# Система умных светофоров

## Описание

Система умных светофоров — это комплексное решение для оптимизации дорожного трафика с использованием машинного обучения и компьютерного зрения. Проект позволяет:

- **Моделировать** транспортные потоки
- **Анализировать видео** с дорожных камер
- **Сравнивать** традиционные и адаптивные контроллеры светофоров
- **Визуализировать** данные и результаты
- **Экспериментировать** с различными стратегиями управления

Цель: уменьшение времени ожидания на светофорах, улучшение пропускной способности и снижение расхода топлива.

---

## Структура проекта

| Файл                        | Описание                                                                 |
| --------------------------- | ------------------------------------------------------------------------ |
| `video_processor.py`        | Анализ видео и обнаружение транспортных средств                          |
| `traffic_model.py`          | Модель транспортного потока и перекрестка                                |
| `traditional_controller.py` | Контроллер с фиксированными фазами                                       |
| `smart_controller.py`       | Адаптивный контроллер с нечеткой логикой, Q-обучением и методом Вебстера |
| `simulation_manager.py`     | Менеджер симуляций и сбор метрик                                         |
| `visualization.py`          | Визуализация и анимация                                                  |
| `main.py`                   | Основной запуск системы                                                  |
| `config_templates.py`       | Шаблоны конфигурационных файлов                                          |
| `tests.py`                  | Юнит-тесты и интеграционное тестирование                                 |

---

## Установка

### Требования:

- Python 3.8+
- pip
- (рекомендуется) виртуальное окружение

### Установка зависимостей:

```bash
pip install -r requirements.txt
```

### Установка PyTorch с поддержкой CUDA (если доступно):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Быстрый старт

### 1. Симуляция с традиционным контроллером

```bash
python main.py --mode simulation --output-dir results
```

### 2. Обработка видео с камеры

```bash
python main.py --mode video --video-source 0 --output-dir results
```

### 3. Сравнение контроллеров

```bash
python main.py --mode comparison --output-dir results
```

### 4. Запуск серии экспериментов

```bash
python main.py --mode experiment --experiment experiments.yaml
```

---

## Конфигурация

Конфигурационные файлы могут быть в формате JSON или YAML. Основные параметры:

### 1. **controllers.traditional**

```yaml
traditional:
  phases:
    - phase_id: 0
      phase_name: North-South Green
      durations: { green: 30, yellow: 5, red: 0 }
      directions: { north: green, south: green, east: red, west: red }
```

### 2. **controllers.smart**

```yaml
smart:
  algorithm: fuzzy
  min_green: 15
  max_green: 90
  yellow: 5
```

### 3. **simulation**

```yaml
simulation:
  time: 300
  time_step: 1.0
  arrival_rates: { north: 10, south: 10, east: 15, west: 15 }
  vehicle_types: { car: 0.6, truck: 0.1, bus: 0.2, motorcycle: 0.1 }
```

### 4. **visualization**

```yaml
visualization:
  enabled: true
  window_size: '800,600'
  scale: 10.0
  fps: 30
  colors:
    background: [255, 255, 255]
    road: [100, 100, 100]
```

### 5. **video**

```yaml
video:
  model:
    path: yolov8n.pt
    confidence_threshold: 0.5
    iou_threshold: 0.45
    classes: ['car', 'truck', 'bus', 'motorcycle']
  roi:
    north: [[100, 200], [300, 400]]
    south: [[100, 200], [300, 400]]
```

---

## Примеры использования

### 1. **Моделирование одного перекрестка**

```python
from smart_controller import SmartController
from traffic_model import Lane, Intersection, Vehicle
from visualization import IntersectionVisualizer

# Создаем умный контроллер
smart_controller = SmartController(
    directions=['north', 'south', 'east', 'west'],
    algorithm='fuzzy'
)

# Создаем полосы
lanes = [
    Lane(lane_id=i, direction=direction, length=100, max_speed=20)
    for i, direction in enumerate(['north', 'south', 'east', 'west'])
]

# Создаем перекресток
intersection = Intersection(lanes=lanes, traffic_light=smart_controller)
```

### 2. **Анализ видео с дорожной камеры**

```bash
# Запуск анализа видео
python main.py --mode video --video-source 3.mp4 --output-dir results
```

### 3. **Сравнение алгоритмов управления**

```bash
# Запуск сравнения
python main.py --mode comparison --output-dir results
```

### 4. **Проведение серии экспериментов**

```bash
# Запуск экспериментов
python main.py --mode experiment --experiment experiments.yaml
```

---

## Разработка

### Архитектура системы

```
[Видео] --> [VideoProcessor] --> [SmartController]
                    ↓
[Симуляция] --> [SimulationManager] --> [MetricsCalculator]
                    ↓
[Visualization] <-- [IntersectionVisualizer]
```

### Ключевые классы

| Класс                    | Описание                            |
| ------------------------ | ----------------------------------- |
| `Vehicle`                | Модель транспортного средства       |
| `Lane`                   | Полоса движения                     |
| `Intersection`           | Перекресток с полосами              |
| `TraditionalController`  | Контроллер с фиксированными фазами  |
| `SmartController`        | Контроллер с адаптивным управлением |
| `SimulationManager`      | Управление симуляцией и сбор метрик |
| `IntersectionVisualizer` | Визуализация перекрестка            |
| `ResultsVisualizer`      | Сравнение результатов               |

### Как добавить новый алгоритм управления:

1. Добавьте метод в `SmartController`
2. Реализуйте логику в `calculate_phase_duration()`
3. Добавьте метод в `--algorithm` в `argparse`
4. Протестируйте с помощью `tests.py`

### Как расширить модель транспортного потока:

1. Добавьте новые типы ТС в `Vehicle`
2. Расширьте логику в `Lane`
3. Добавьте новые параметры в `TrafficSimulation`
4. Обновите конфигурации в `config_templates.py`

---

## Тестирование

### Запуск юнит-тестов:

```bash
python tests.py --test-module all
```

### Тестирование производительности:

```bash
python tests.py --performance --iterations 100
```

### Интеграционное тестирование:

```bash
python tests.py --test-module integration
```

---

## Результаты

### Собираемые метрики:

- **Среднее время ожидания** (секунды)
- **Максимальная длина очереди** (ТС)
- **Пропускная способность** (ТС/минута)
- **Расход топлива** (литры)

### Пример отчета:

```
Традиционный контроллер:
- average_waiting_time: 45.2 сек
- max_queue_length: 9 ТС
- throughput: 12 ТС/мин

Умный контроллер:
- average_waiting_time: 28.7 сек
- max_queue_length: 4 ТС
- throughput: 20 ТС/мин
```

---

## Ограничения и известные проблемы

| Ограничение                       | Описание                                                           |
| --------------------------------- | ------------------------------------------------------------------ |
| Зависимость от качества видео     | Низкое качество видео может привести к ошибкам обнаружения         |
| Высокие требования к GPU          | Для обработки видео в реальном времени нужен мощный GPU            |
| Ограниченные типы ТС              | На данный момент поддерживаются только car, truck, bus, motorcycle |
| Требуется настройка параметров    | Алгоритмы требуют настройки под конкретные перекрестки             |
| Ограниченная поддержка алгоритмов | Нужно расширять поддержку новых методов (например, нейросети)      |
| Точность YOLO                     | Модель YOLOv8 может ошибаться на высоком трафике                   |
| Поддержка Windows                 | Нужно тестировать работу с OpenCV и видеокамерами                  |
| Визуализация                      | Не все метрики отображаются на графиках                            |
| Обучение с подкреплением          | Требует длительной настройки и обучения                            |

---

## Помощь и сообщение об ошибках

Если вы нашли баг, хотите предложить улучшение или запросить поддержку:

- **Email:** project@example.com
- **Telegram:** @smart_traffic_system
- **GitHub Issues:** [issues](https://github.com/smart-traffic-system/issues)
- **Документация:** [wiki](https://github.com/smart-traffic-system/wiki)

---

### 📌 Итог

Эти файлы:

- `requirements.txt` — обеспечивает детерминированную установку зависимостей
- `README.md` — содержит полную документацию, примеры и описание архитектуры
- `main.py` — позволяет запускать систему в разных режимах
- `tests.py` — обеспечивает надежность и стабильность

Теперь вы можете:

- Установить систему
- Запустить симуляцию
- Проанализировать видео
- Сравнить контроллеры
- Собрать отчет с метриками
- Расширять систему новыми алгоритмами

Проект готов к использованию и развитию.
