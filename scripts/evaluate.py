"""
Финальная оценка и сравнение всех моделей.

Загружает сохранённые метрики всех экспериментов (baseline, улучшенный baseline,
HOG+SVM), оценивает модели на тестовой выборке и генерирует итоговый отчёт
с визуализацией результатов.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def evaluate_yolo_on_test(
    model_path: str,
    yaml_path: str,
    experiment_name: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> dict:
    """
    Оценивает YOLO-модель на тестовой выборке датасета.

    Args:
        model_path: Путь к файлу весов (.pt).
        yaml_path: Путь к YAML-конфигу датасета.
        experiment_name: Название эксперимента для отчёта.
        conf_threshold: Порог уверенности при инференсе.
        iou_threshold: Порог IoU для расчёта mAP.

    Returns:
        Словарь с метриками качества на test-split.
    """
    print(f"\nОценка: {experiment_name}")
    model = YOLO(model_path)
    results = model.val(
        data=yaml_path,
        split="test",
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    p = float(results.box.mp)
    r = float(results.box.mr)
    return {
        "experiment": experiment_name,
        "mAP50": round(float(results.box.map50), 4),
        "mAP50-95": round(float(results.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r + 1e-9), 4),
    }


def load_custom_metrics(metrics_path: str, experiment_name: str) -> dict:
    """
    Загружает метрики пользовательского детектора из файла.

    Args:
        metrics_path: Путь к файлу с метриками (txt формат key: value).
        experiment_name: Название эксперимента.

    Returns:
        Словарь с метриками.
    """
    metrics = {"experiment": experiment_name}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line and not line.startswith("="):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                if key in ("mAP50", "mAP50-95", "precision", "recall", "f1"):
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
    return metrics


def generate_markdown_report(all_metrics: list, output_path: str) -> None:
    """
    Генерирует Markdown-отчёт со сравнительной таблицей и выводами.

    Args:
        all_metrics: Список словарей с метриками всех экспериментов.
        output_path: Путь для сохранения .md файла.
    """
    lines = [
        "# Отчёт: Детекция номерных знаков",
        "",
        "## Описание задачи",
        "",
        "**Датасет:** [Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)  ",
        "**Класс:** `licence` (номерной знак автомобиля)  ",
        "**Задача:** Object Detection (обнаружение объектов)",
        "",
        "## Обоснование выбора задачи и датасета",
        "",
        "Детекция номерных знаков — востребованная практическая задача:",
        "парковочные системы, контроль доступа, видеонаблюдение, автоматизация штрафов ГИБДД.",
        "Датасет содержит 433 реальных фотографии автомобилей с разметкой в формате Pascal VOC.",
        "",
        "## Обоснование метрик",
        "",
        "| Метрика | Обоснование |",
        "|---------|-------------|",
        "| **mAP@0.5** | Стандарт для задач детекции; учитывает и точность боксов, и уверенность |",
        "| **mAP@0.5:0.95** | Строгая оценка качества локализации при разных порогах IoU |",
        "| **Precision** | Доля верных детекций среди всех детекций (важно для ложных срабатываний) |",
        "| **Recall** | Доля найденных знаков среди всех реальных (важно для пропусков) |",
        "| **F1** | Гармоническое среднее Precision и Recall |",
        "",
        "В задаче важен баланс Precision и Recall: пропущенный номерной знак",
        "так же нежелателен, как и ложное срабатывание.",
        "",
        "## Результаты",
        "",
        "### Сравнительная таблица (Test split)",
        "",
        "| Эксперимент | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
        "|-------------|---------|--------------|-----------|--------|----|",
    ]

    for m in all_metrics:
        row = (
            f"| {m.get('experiment', '—')} "
            f"| {m.get('mAP50', 0):.4f} "
            f"| {m.get('mAP50-95', 0):.4f} "
            f"| {m.get('precision', 0):.4f} "
            f"| {m.get('recall', 0):.4f} "
            f"| {m.get('f1', 0):.4f} |"
        )
        lines.append(row)


    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Отчёт сохранён: {output_path}")


def print_comparison_table(all_metrics: list) -> None:
    """
    Выводит сравнительную таблицу в консоль.

    Args:
        all_metrics: Список словарей с метриками.
    """
    print("\n" + "=" * 75)
    print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (Test split)")
    print("=" * 75)
    fmt = "{:<28} {:>8} {:>12} {:>11} {:>8} {:>8}"
    print(fmt.format("Эксперимент", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"))
    print("-" * 75)
    for m in all_metrics:
        print(fmt.format(
            m.get("experiment", "—")[:27],
            f"{m.get('mAP50', 0):.4f}",
            f"{m.get('mAP50-95', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
        ))
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Финальная оценка и сравнение моделей")
    parser.add_argument("--yaml", default="configs/car_plate.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--report-path", default="report/final_report.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_metrics = []

    # Оцениваем baseline YOLOv11n
    baseline_weights = results_dir / "baseline_yolo11n" / "weights" / "best.pt"
    if baseline_weights.exists():
        m = evaluate_yolo_on_test(str(baseline_weights), args.yaml, "YOLOv11n (baseline)")
        all_metrics.append(m)
    else:
        print(f"[WARN] Веса baseline не найдены: {baseline_weights}")

    # Оцениваем эксперименты с гипотезами
    for hyp_name in ["H1_yolo11s", "H2_augmentations", "H3_imgsz832", "H4_cosine_lr"]:
        weights = results_dir / hyp_name / "weights" / "best.pt"
        if weights.exists():
            m = evaluate_yolo_on_test(str(weights), args.yaml, hyp_name)
            all_metrics.append(m)

    # Оцениваем финальный улучшенный конфиг
    best_weights = results_dir / "improved_best" / "weights" / "best.pt"
    if best_weights.exists():
        m = evaluate_yolo_on_test(str(best_weights), args.yaml, "YOLOv11s (improved_best)")
        all_metrics.append(m)

    # Загружаем метрики HOG+SVM
    custom_metrics_file = results_dir / "custom" / "metrics_summary.txt"
    if custom_metrics_file.exists():
        m = load_custom_metrics(str(custom_metrics_file), "HOG+SVM (custom)")
        all_metrics.append(m)
    else:
        print(f"[INFO] Метрики HOG+SVM не найдены. Запустите custom_detector.py --mode both")

    if not all_metrics:
        print("[ERROR] Нет метрик для сравнения. Сначала обучите модели.")
        exit(1)

    print_comparison_table(all_metrics)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    generate_markdown_report(all_metrics, args.report_path)

    # Сохраняем метрики в JSON
    json_path = str(Path(args.report_path).with_suffix(".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"JSON-метрики: {json_path}")
