"""
Улучшенный baseline для детекции номерных знаков.

Гипотезы для улучшения качества:
  H1: Более крупная модель (YOLOv11s вместо YOLOv11n) → лучшая точность.
  H2: Агрессивные аугментации (random crop, perspective, mosaic) → устойчивость к условиям съёмки.
  H3: Увеличение разрешения входа (640 → 832) → лучшее обнаружение мелких объектов.
  H4: Подбор lr-scheduler (cosine annealing) → более стабильная сходимость.

Запускается последовательно для каждой гипотезы, затем обучается
"лучший" конфиг, объединяющий все подтверждённые улучшения.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def train_experiment(
    yaml_path: str,
    config: dict,
    project: str,
    name: str,
    device: str = "0"
) -> dict:
    """
    Запускает один обучающий эксперимент с заданной конфигурацией.

    Args:
        yaml_path: Путь к YAML-конфигу датасета.
        config: Словарь гиперпараметров обучения.
        project: Директория для сохранения результатов.
        name: Имя эксперимента.
        device: Устройство ('cpu', '0').

    Returns:
        Словарь с метриками на валидационной выборке.
    """
    print(f"\n{'='*60}")
    print(f"Эксперимент: {name}")
    print(f"{'='*60}")

    model_weights = config.pop("model", "yolo11n.pt")
    model = YOLO(model_weights)

    model.train(
        data=yaml_path,
        project=project,
        name=name,
        device=device,
        verbose=True,
        plots=True,
        **config
    )

    val_results = model.val(data=yaml_path, split="val")

    p = float(val_results.box.mp)
    r = float(val_results.box.mr)
    metrics = {
        "model": model_weights,
        "experiment": name,
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r + 1e-9), 4),
    }

    print(f"\nРезультаты {name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    _save_metrics(metrics, project, name)
    return metrics


def run_hypothesis_experiments(
    yaml_path: str,
    base_epochs: int,
    project: str,
    device: str
) -> list:
    """
    Последовательно проверяет все гипотезы улучшения бейзлайна.

    Каждая гипотеза меняет ровно один параметр относительно baseline,
    что позволяет оценить вклад каждого улучшения изолированно.

    Args:
        yaml_path: Путь к YAML-конфигу датасета.
        base_epochs: Базовое число эпох для каждого эксперимента.
        project: Директория для сохранения результатов.
        device: Устройство обучения.

    Returns:
        Список словарей с метриками для каждой гипотезы.
    """
    results = []

    # H1: Модель большего размера (YOLOv11s)
    h1_config = {
        "model": "yolo11s.pt",
        "epochs": base_epochs,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    results.append(train_experiment(yaml_path, h1_config, project, "H1_yolo11s", device))

    # H2: Агрессивные аугментации (YOLOv11n + augmentations)
    h2_config = {
        "model": "yolo11n.pt",
        "epochs": base_epochs,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Усиленные аугментации
        "hsv_h": 0.05,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 5.0,           # Небольшой поворот
        "perspective": 0.001,     # Перспективные искажения
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,             # MixUp аугментация
        "copy_paste": 0.1,        # Copy-Paste для малых объектов
    }
    results.append(train_experiment(yaml_path, h2_config, project, "H2_augmentations", device))

    # H3: Увеличенное входное разрешение (832)
    h3_config = {
        "model": "yolo11n.pt",
        "epochs": base_epochs,
        "imgsz": 832,             # Выше разрешение → лучше для мелких объектов
        "batch": 8,               # Меньший батч из-за памяти
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    results.append(train_experiment(yaml_path, h3_config, project, "H3_imgsz832", device))

    # H4: Cosine LR + более долгое обучение
    h4_config = {
        "model": "yolo11n.pt",
        "epochs": base_epochs,
        "imgsz": 640,
        "batch": 16,
        "lr0": 0.01,
        "lrf": 0.001,             # Более мягкое снижение lr (cosine)
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "cos_lr": True,           # Cosine annealing
        "warmup_epochs": 5,       # Warmup
        "mosaic": 1.0,
        "mixup": 0.0,
    }
    results.append(train_experiment(yaml_path, h4_config, project, "H4_cosine_lr", device))

    return results


def train_best_config(
    yaml_path: str,
    epochs: int,
    project: str,
    device: str
) -> dict:
    """
    Обучает финальную улучшенную модель, объединяющую все подтверждённые гипотезы.

    Финальная конфигурация:
    - YOLOv11s (H1): более мощная архитектура
    - Агрессивные аугментации (H2): устойчивость к условиям
    - Разрешение 832 (H3): лучшее обнаружение
    - Cosine LR + warmup (H4): стабильная сходимость

    Args:
        yaml_path: Путь к YAML-конфигу датасета.
        epochs: Число эпох обучения.
        project: Директория для сохранения результатов.
        device: Устройство обучения.

    Returns:
        Словарь с метриками финальной модели.
    """
    print(f"\n{'='*60}")
    print("ФИНАЛЬНАЯ УЛУЧШЕННАЯ МОДЕЛЬ (все гипотезы)")
    print(f"{'='*60}")

    best_config = {
        "model": "yolo11s.pt",       # H1
        "epochs": epochs,
        "imgsz": 832,                # H3
        "batch": 8,
        "lr0": 0.01,
        "lrf": 0.001,                # H4
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "cos_lr": True,              # H4
        "warmup_epochs": 5,          # H4
        # Аугментации (H2)
        "hsv_h": 0.05,
        "hsv_s": 0.9,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "perspective": 0.001,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
    }

    return train_experiment(yaml_path, best_config, project, "improved_best", device)


def _save_metrics(metrics: dict, project: str, name: str) -> None:
    """
    Сохраняет метрики эксперимента в файл.

    Args:
        metrics: Словарь с метриками.
        project: Директория проекта.
        name: Имя эксперимента.
    """
    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics_summary.txt"
    with open(out_file, "w") as f:
        f.write("=== Metrics Summary ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def compare_experiments(all_metrics: list, save_path: str) -> None:
    """
    Сравнивает результаты всех экспериментов и сохраняет сводную таблицу.

    Args:
        all_metrics: Список словарей с метриками экспериментов.
        save_path: Путь для сохранения сводной таблицы.
    """
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)
    header = f"{'Эксперимент':<30} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * 70)

    lines = [header, "-" * 70]
    for m in all_metrics:
        row = (
            f"{m.get('experiment', m.get('model', '')):<30}"
            f"{m['mAP50']:>8.4f}"
            f"{m['mAP50-95']:>10.4f}"
            f"{m['precision']:>10.4f}"
            f"{m['recall']:>8.4f}"
            f"{m['f1']:>8.4f}"
        )
        print(row)
        lines.append(row)

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nСводная таблица сохранена: {save_path}")

    # JSON для машиночитаемости
    json_path = save_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Улучшение baseline + проверка гипотез")
    parser.add_argument("--yaml", default="configs/car_plate.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--project", default="results")
    parser.add_argument("--device", default="0")
    parser.add_argument("--skip-hypotheses", action="store_true",
                        help="Пропустить отдельные эксперименты, обучить только лучший конфиг")
    args = parser.parse_args()

    all_metrics = []

    if not args.skip_hypotheses:
        print("=== Проверка гипотез ===")
        hypothesis_results = run_hypothesis_experiments(
            yaml_path=args.yaml,
            base_epochs=args.epochs,
            project=args.project,
            device=args.device
        )
        all_metrics.extend(hypothesis_results)

    print("\n=== Финальный улучшенный конфиг ===")
    best_metrics = train_best_config(
        yaml_path=args.yaml,
        epochs=args.epochs,
        project=args.project,
        device=args.device
    )
    best_metrics["experiment"] = "improved_best"
    all_metrics.append(best_metrics)

    compare_experiments(all_metrics, f"{args.project}/experiment_comparison.txt")
