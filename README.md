# Лабораторная работа 1: Детекция номерных знаков

**Курс:** Киберфизические системы
**Задача:** Обнаружение и распознавание объектов (Object Detection)
**Вариант:** На тройку (Проведение исследований моделями обнаружения и распознавания объектов)  
**Датасет:** [Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)  
**Стек:** Python · YOLOv11 (ultralytics)

---

## Описание задачи

**Бизнес-контекст:** Автоматическое обнаружение номерных знаков автомобилей применяется в:
- Системах автоматической парковки и контроля доступа
- Видеонаблюдении и фиксации нарушений ПДД
- Умных городских системах управления трафиком

**Датасет:** 433 изображения автомобилей с разметкой bounding box в формате Pascal VOC (XML).  
**Единственный класс:** `licence` (номерной знак).

---

## Метрики качества

| Метрика | Описание |
|---------|----------|
| **mAP@0.5** | Mean Average Precision при IoU ≥ 0.5. Основная метрика детекции |
| **mAP@0.5:0.95** | mAP усреднённый по порогам IoU 0.5–0.95. Строже оценивает локализацию |
| **Precision** | Точность: доля правильных детекций среди всех детекций |
| **Recall** | Полнота: доля найденных номерных знаков среди всех реальных |
| **F1** | Гармоническое среднее Precision и Recall |

**Обоснование:** mAP@0.5 — стандарт для задач детекции (COCO, Pascal VOC benchmark).
Precision и Recall важны раздельно: в охранных системах критичен высокий Recall
(нельзя пропустить), в системах оплаты — высокий Precision (нельзя ошибиться).

---

## Структура проекта

```
car-plate-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── car_plate.yaml          # YOLO-конфиг датасета (генерируется автоматически)
├── scripts/
│   ├── prepare_dataset.py      # Конвертация XML→YOLO, разбивка на сплиты
│   ├── train_baseline.py       # Обучение baseline YOLOv11n
│   ├── train_improved.py       # Гипотезы + улучшенный baseline
│   ├── custom_detector.py      # Обучение YOLOv11 с нуля (pretrained=False)
│   ├── evaluate.py             # Финальная оценка + отчёт
│   └── run_all.py              # Полный пайплайн одной командой
├── dataset/                    # Создаётся при запуске prepare_dataset.py
│   ├── train/images/ & labels/
│   ├── val/images/ & labels/
│   └── test/images/ & labels/
├── results/                    # Веса моделей, графики обучения
└── report/                     # Итоговый Markdown-отчёт
```

---

## Установка и запуск

### Требования

- Python 3.9+
- pip
- (Опционально) NVIDIA GPU + CUDA 11.8+ для ускорения обучения

### 1. Клонирование репозитория

```bash
git clone https://github.com/YOUR_USERNAME/car-plate-detection.git
cd car-plate-detection
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Загрузка датасета

**Способ 1 — через Kaggle CLI:**
```bash
pip install kaggle
# Поместите kaggle.json в ~/.kaggle/ (Linux/macOS) или %USERPROFILE%/.kaggle/ (Windows)
kaggle datasets download -d andrewmvd/car-plate-detection
unzip car-plate-detection.zip -d raw_data
```

**Способ 2 — вручную:**  
Скачайте с [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) и распакуйте.  
Ожидаемая структура:
```
raw_data/
├── images/          # *.png изображения
└── annotations/     # *.xml аннотации (Pascal VOC)
```

---

## Запуск

### Вариант A: Полный пайплайн одной командой

Запускает все шаги последовательно: подготовка данных → baseline → гипотезы → обучение с нуля → отчёт.

```bash
# GPU (рекомендуется)
python scripts/run_all.py \
    --images raw_data/images \
    --annotations raw_data/annotations \
    --epochs 50 \
    --device 0

# CPU (медленнее, для тестирования)
python scripts/run_all.py \
    --images raw_data/images \
    --annotations raw_data/annotations \
    --epochs 10 \
    --batch 8 \
    --device cpu \
    --skip-hypotheses
```

После завершения:
- **Отчёт:** `report/final_report.md`
- **Веса:** `results/*/weights/best.pt`
- **Графики:** `results/*/`

---

### Вариант B: Пошаговый запуск

#### Шаг 1: Подготовка датасета

```bash
python scripts/prepare_dataset.py \
    --images raw_data/images \
    --annotations raw_data/annotations \
    --output dataset \
    --yaml configs/car_plate.yaml
```

Разбивка: 70% train / 20% val / 10% test.

#### Шаг 2: Baseline YOLOv11n

```bash
python scripts/train_baseline.py \
    --yaml configs/car_plate.yaml \
    --epochs 50 \
    --device 0
```

Результаты: `results/baseline_yolo11n/`

#### Шаг 3: Улучшение baseline

Проверяются 4 гипотезы (H1–H4), затем обучается финальный конфиг:

```bash
python scripts/train_improved.py \
    --yaml configs/car_plate.yaml \
    --epochs 50 \
    --device 0

# Только финальный конфиг (без отдельных гипотез):
python scripts/train_improved.py \
    --yaml configs/car_plate.yaml \
    --epochs 50 \
    --device 0 \
    --skip-hypotheses
```

#### Шаг 4: Имплементация с нуля (pretrained=False)

```bash
python scripts/custom_detector.py \
    --yaml configs/car_plate.yaml \
    --epochs 100 \
    --device 0 \
    --variants yolo11n yolo11s
```

#### Шаг 5: Финальный отчёт

```bash
python scripts/evaluate.py \
    --yaml configs/car_plate.yaml \
    --results-dir results \
    --report-path report/final_report.md
```

---

## Архитектура решения

### YOLOv11 (ultralytics)

YOLOv11 — однопроходный (one-stage) детектор. Принимает изображение целиком,
предсказывает bounding boxes и классы для всех объектов за один forward pass.
Предобученные веса (COCO) обеспечивают сильный transfer learning даже на малых датасетах.

| Модель | Параметры | Назначение |
|--------|-----------|------------|
| YOLOv11n | ~2.6M | Baseline (минимальная) |
| YOLOv11s | ~9.4M | Улучшенный baseline |

### Шаг 4: YOLOv11 обучение с нуля (`pretrained=False`)

"Самостоятельная имплементация" в рамках стека ultralytics — обучение той же архитектуры **без предобученных весов**:

```python
YOLO("yolo11n.yaml")  # .yaml = архитектура, случайная инициализация весов
YOLO("yolo11n.pt")    # .pt   = архитектура + веса COCO (baseline/improved)
```

Обучаются две архитектуры (`yolo11n`, `yolo11s`) с нуля с теми же аугментациями, что и в шаге 3. Это позволяет изолировать вклад transfer learning и ответить на вопрос: **насколько важны предобученные веса при датасете в 433 изображения?**

---

## Гипотезы улучшения baseline

| # | Гипотеза | Изменение | Ожидаемый эффект |
|---|----------|-----------|-----------------|
| H1 | Более крупная модель | YOLOv11n → YOLOv11s | +mAP за счёт ёмкости |
| H2 | Агрессивные аугментации | +perspective, +copy_paste, +mixup | Устойчивость к ракурсу и освещению |
| H3 | Высокое разрешение | 640 → 832 px | Лучше детектирует мелкие знаки |
| H4 | Cosine LR + warmup | cos_lr=True, warmup_epochs=5 | Стабильнее сходимость |

---

# Отчёт: Детекция номерных знаков

## Описание задачи

**Датасет:** [Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)  
**Класс:** `licence` (номерной знак автомобиля)  
**Задача:** Object Detection (обнаружение объектов)

## Обоснование выбора задачи и датасета

Детекция номерных знаков — востребованная практическая задача:
парковочные системы, контроль доступа, видеонаблюдение, автоматизация штрафов ГИБДД.
Датасет содержит 433 реальных фотографии автомобилей с разметкой в формате Pascal VOC.

## Обоснование метрик

| Метрика | Обоснование |
|---------|-------------|
| **mAP@0.5** | Стандарт для задач детекции; учитывает и точность боксов, и уверенность |
| **mAP@0.5:0.95** | Строгая оценка качества локализации при разных порогах IoU |
| **Precision** | Доля верных детекций среди всех детекций (важно для ложных срабатываний) |
| **Recall** | Доля найденных знаков среди всех реальных (важно для пропусков) |
| **F1** | Гармоническое среднее Precision и Recall |

В задаче важен баланс Precision и Recall: пропущенный номерной знак
так же нежелателен, как и ложное срабатывание.

## Результаты

### Сравнительная таблица (Test split)

| Эксперимент | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|-------------|---------|--------------|-----------|--------|----|
| YOLOv11n baseline (pretrained) | 0.9364 | 0.5657 | 0.9506 | 0.8710 | 0.9091 |
| H1_yolo11s | 0.9332 | 0.5605 | 0.9557 | 0.8602 | 0.9055 |
| H2_augmentations | 0.9339 | 0.5505 | 0.9644 | 0.8743 | 0.9172 |
| H3_imgsz832 | 0.9302 | 0.5841 | 0.9401 | 0.8925 | 0.9157 |
| H4_cosine_lr | 0.9172 | 0.5759 | 0.9208 | 0.8817 | 0.9008 |
| YOLOv11s improved (pretrained) | 0.9496 | 0.5780 | 0.9655 | 0.8817 | 0.9217 |
| yolo11n from scratch | 0.8951 | 0.4453 | 0.9347 | 0.7957 | 0.8596 |
| yolo11s from scratch | 0.8870 | 0.4813 | 0.9457 | 0.7527 | 0.8382 |

### Анализ гипотез

1. **Одиночные изменения (H1-H4):** По отдельности гипотезы не дали ощутимого прироста основной метрики `mAP@0.5` относительно сильного baseline. Однако они улучшили отдельные характеристики:
   - **H2 (Аугментации):** Снизили переобучение и обеспечили высокую точность (`Precision` = 0.9644).
   - **H3 (Разрешение 832):** Значительно улучшило качество строгой локализации рамок (`mAP@0.5:0.95` вырос с 0.5657 до 0.5841) и полноту (`Recall` = 0.8925) за счет лучшей видимости мелких номеров.
2. **Синергетический эффект (improved_best):** Объединение всех гипотез дало лучший результат, скомпенсировав недостатки друг друга. Комплексный подход превзошел baseline: `mAP@0.5` вырос на **+0.0132** (до 0.9496), а `F1`-score достиг максимума в **0.9217**.

### Выводы по обучению с нуля (From Scratch)

1. **Критичность Transfer Learning:** Отказ от предобученных весов (модели "from scratch") привел к серьезному падению метрик (снижение `mAP@0.5` на ~4-5%). На небольшом датасете из 433 изображений нейросети крайне сложно самостоятельно выучить базовые визуальные признаки.
2. **Риск переобучения (Overfitting):** При обучении с нуля более тяжелая модель `yolo11s` (0.8870) уступила более легкой `yolo11n` (0.8951). Отсутствие Transfer Learning вкупе с большой ёмкостью сети привело к её переобучению на тренировочной выборке.

### Итог
Улучшенный пайплайн на базе предобученной YOLOv11s отлично справился с задачей, достигнув 95% mAP@0.5 и ~96.5% Precision. Такого качества детекции более чем достаточно для применения в системах контроля доступа и умных парковок.

