"""
Обучение базовой (baseline) модели YOLOv11n для детекции номерных знаков.

Baseline — минимальная конфигурация без дополнительных улучшений:
модель YOLOv11n (nano), стандартные гиперпараметры, базовые аугментации.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_baseline(
    yaml_path: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "results",
    name: str = "baseline_yolo11n",
    device: str = "cpu"
) -> dict:
    """
    Обучает baseline-модель YOLOv11n на датасете Car Plate Detection.

    Baseline-конфигурация намеренно минималистична: используется наименьшая
    модель семейства YOLOv11 (nano) без дополнительных аугментаций и тюнинга
    гиперпараметров. Это позволяет получить точку отсчёта для последующих улучшений.

    Args:
        yaml_path: Путь к YAML-конфигу датасета.
        epochs: Количество эпох обучения.
        imgsz: Размер входного изображения (пиксели).
        batch: Размер батча.
        project: Директория для сохранения результатов.
        name: Имя эксперимента.
        device: Устройство обучения ('cpu', '0', 'cuda').

    Returns:
        Словарь с метриками на валидационной выборке:
        mAP50, mAP50-95, precision, recall.
    """
    print("=" * 60)
    print("BASELINE: YOLOv11n — стандартная конфигурация")
    print("=" * 60)

    # Загружаем предобученную модель YOLOv11n
    model = YOLO("yolo11n.pt")

    # Обучение с минимальными настройками (baseline)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        # Стандартные гиперпараметры (без улучшений)
        lr0=0.01,           # Начальный learning rate
        lrf=0.01,           # Конечный lr (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        # Минимальные аугментации
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        verbose=True,
        plots=True,
    )

    # Валидация на val-split
    val_results = model.val(data=yaml_path, split="val")

    metrics = {
        "model": "YOLOv11n (baseline)",
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(float(val_results.box.mp), 4),
        "recall": round(float(val_results.box.mr), 4),
    }
    metrics["f1"] = round(
        2 * metrics["precision"] * metrics["recall"] /
        (metrics["precision"] + metrics["recall"] + 1e-9), 4
    )

    print("\n=== Метрики Baseline ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Сохраняем метрики
    _save_metrics(metrics, project, name)

    return metrics


def _save_metrics(metrics: dict, project: str, name: str) -> None:
    """
    Сохраняет метрики в текстовый файл.

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
    print(f"Метрики сохранены: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение baseline YOLOv11n")
    parser.add_argument("--yaml", default="configs/car_plate.yaml", help="Путь к YAML-конфигу")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="results")
    parser.add_argument("--device", default="0", help="'cpu' или '0' для GPU")
    args = parser.parse_args()

    metrics = train_baseline(
        yaml_path=args.yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        device=args.device
    )
