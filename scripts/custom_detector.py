"""
Шаг 4: Имплементация алгоритма машинного обучения (ultralytics).

Согласно заданию на тройку, весь стек — ultralytics (YOLOv11).
"Самостоятельная имплементация" здесь — обучение модели с нуля
(pretrained=False), без заимствования весов с COCO.

Это позволяет честно ответить на вопрос:
"Что именно даёт transfer learning при малом датасете?"

Обучаются две архитектуры из семейства YOLOv11 без претренинга:
  - yolo11n (nano)  — та же архитектура, что и в baseline (шаг 2)
  - yolo11s (small) — та же архитектура, что в улучшенном baseline (шаг 3)

Результаты сравниваются с шагами 2 и 3 по тем же метрикам:
mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1.
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def train_from_scratch(
    yaml_path: str,
    model_variant: str = "yolo11n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "results",
    name: str = "custom_scratch",
    device: str = "0"
) -> dict:
    """
    Обучает YOLOv11 с нуля без использования предобученных весов.

    Ключевое отличие от шагов 2-3: передаётся YAML-конфиг архитектуры
    (не .pt файл с весами), что инициализирует сеть случайными весами
    вместо весов, предобученных на COCO.

    Параметр pretrained=False явно отключает загрузку весов даже если
    ultralytics попытается их скачать автоматически.

    Без предобученных весов сеть вынуждена научиться всему с нуля
    по 433 изображениям — отсюда ожидаемо более низкие метрики.
    Это и демонстрирует ценность transfer learning для малых датасетов.

    Args:
        yaml_path: Путь к YAML-конфигу датасета.
        model_variant: Архитектура — 'yolo11n', 'yolo11s', 'yolo11m'.
        epochs: Количество эпох (рекомендуется ≥100 без претренинга).
        imgsz: Размер входного изображения.
        batch: Размер батча.
        project: Директория для сохранения результатов.
        name: Имя эксперимента.
        device: Устройство ('cpu', '0' для GPU).

    Returns:
        Словарь с метриками на валидационной выборке:
        experiment, model, mAP50, mAP50-95, precision, recall, f1.
    """
    print("=" * 60)
    print(f"ШАГ 4: {model_variant.upper()} — обучение С НУЛЯ (pretrained=False)")
    print("=" * 60)

    # Используем .yaml конфиг архитектуры — без весов (случайная инициализация).
    # YOLO("yolo11n.yaml") vs YOLO("yolo11n.pt"):
    #   .yaml — только структура сети, веса Xavier/He инициализация
    #   .pt   — структура + предобученные на COCO веса
    model = YOLO(f"{model_variant}.yaml")

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        pretrained=False,       # явное отключение претренинга
        # Чуть более высокий lr: без претренинга нужен больший начальный шаг
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,
        # Аугментации из улучшенного baseline (шаг 3) — оставляем те же,
        # чтобы разница метрик отражала именно вклад предобученных весов
        hsv_h=0.05,
        hsv_s=0.9,
        hsv_v=0.5,
        degrees=5.0,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        verbose=True,
        plots=True,
    )

    val_results = model.val(data=yaml_path, split="val")

    p = float(val_results.box.mp)
    r = float(val_results.box.mr)
    metrics = {
        "experiment": f"{model_variant} from scratch",
        "model": f"{model_variant}.yaml (pretrained=False)",
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50-95": round(float(val_results.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r + 1e-9), 4),
    }

    print(f"\n=== Метрики: {model_variant} from scratch ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    _save_metrics(metrics, project, name)
    return metrics


def compare_with_pretrained(
    scratch_metrics: list,
    pretrained_results_dir: str,
    project: str
) -> None:
    """
    Сравнивает модели "с нуля" (шаг 4) с предобученными (шаги 2-3).

    Загружает сохранённые метрики baseline и improved_best,
    добавляет результаты обучения с нуля и выводит сводную таблицу.

    Args:
        scratch_metrics: Список метрик моделей, обученных с нуля.
        pretrained_results_dir: Директория с результатами шагов 2-3.
        project: Директория для сохранения итогового сравнения.
    """
    all_metrics = []

    # Загружаем метрики baseline (шаг 2)
    baseline_file = (
        Path(pretrained_results_dir) / "baseline_yolo11n" / "metrics_summary.txt"
    )
    if baseline_file.exists():
        m = _load_metrics_txt(str(baseline_file))
        m["experiment"] = "YOLOv11n baseline (pretrained)"
        all_metrics.append(m)

    # Загружаем метрики улучшенного baseline (шаг 3)
    improved_file = (
        Path(pretrained_results_dir) / "improved_best" / "metrics_summary.txt"
    )
    if improved_file.exists():
        m = _load_metrics_txt(str(improved_file))
        m["experiment"] = "YOLOv11s improved (pretrained)"
        all_metrics.append(m)

    # Добавляем модели с нуля (шаг 4)
    all_metrics.extend(scratch_metrics)

    # Выводим сравнительную таблицу
    print("\n" + "=" * 75)
    print("СРАВНЕНИЕ ШАГОВ 2-4: Pretrained vs From Scratch")
    print("=" * 75)
    fmt = "{:<35} {:>8} {:>12} {:>10} {:>8}"
    print(fmt.format("Эксперимент", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"))
    print("-" * 75)
    for m in all_metrics:
        print(fmt.format(
            m.get("experiment", "—")[:34],
            f"{m.get('mAP50', 0):.4f}",
            f"{m.get('mAP50-95', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
        ))
    print("=" * 75)

    # Вывод о пользе transfer learning
    pretrained = [m for m in all_metrics if "pretrained" in m.get("experiment", "")]
    scratch = [m for m in all_metrics if "scratch" in m.get("experiment", "")]
    if pretrained and scratch:
        best_pre = max(pretrained, key=lambda m: m.get("mAP50", 0))
        best_scr = max(scratch, key=lambda m: m.get("mAP50", 0))
        delta = best_pre["mAP50"] - best_scr["mAP50"]
        print(f"\nВывод: Transfer learning даёт прирост +{delta:.4f} mAP@0.5")
        print("При малом датасете (433 изображения) предобученные веса критичны.")

    # Сохраняем сравнение в JSON
    out_path = Path(project) / "step4_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nСравнение сохранено: {out_path}")


def _save_metrics(metrics: dict, project: str, name: str) -> None:
    """
    Сохраняет метрики эксперимента в текстовый файл.

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


def _load_metrics_txt(path: str) -> dict:
    """
    Загружает метрики из текстового файла формата 'key: value'.

    Args:
        path: Путь к файлу метрик.

    Returns:
        Словарь с метриками (числовые значения приведены к float).
    """
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("="):
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = val
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Шаг 4: YOLOv11 с нуля (ultralytics, pretrained=False)"
    )
    parser.add_argument("--yaml", default="configs/car_plate.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Эпохи обучения (рекомендуется ≥100 для scratch)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="results")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--variants", nargs="+", default=["yolo11n", "yolo11s"],
        help="Архитектуры для обучения с нуля (default: yolo11n yolo11s)"
    )
    args = parser.parse_args()

    scratch_results = []
    for variant in args.variants:
        metrics = train_from_scratch(
            yaml_path=args.yaml,
            model_variant=variant,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=f"custom_scratch_{variant}",
            device=args.device,
        )
        scratch_results.append(metrics)

    compare_with_pretrained(
        scratch_metrics=scratch_results,
        pretrained_results_dir=args.project,
        project=args.project
    )
