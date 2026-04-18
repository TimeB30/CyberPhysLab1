"""
Главный скрипт запуска всего пайплайна лабораторной работы.

Последовательно выполняет все шаги задания на тройку (пункты 2-4),
используя исключительно ultralytics (YOLOv11):

  1. Подготовка датасета (XML → YOLO формат)
  2. Обучение baseline (YOLOv11n, pretrained=True)
  3. Проверка гипотез + улучшенный baseline (YOLOv11s, pretrained=True)
  4. Имплементация с нуля (YOLOv11n/s, pretrained=False)
  5. Финальное сравнение и генерация отчёта

Использование:
    python scripts/run_all.py --images /path/to/images --annotations /path/to/annotations
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_dataset import prepare_dataset, generate_yaml
from train_baseline import train_baseline
from train_improved import run_hypothesis_experiments, train_best_config, compare_experiments
from custom_detector import train_from_scratch, compare_with_pretrained
from evaluate import evaluate_yolo_on_test, generate_markdown_report, print_comparison_table


def run_pipeline(
    images_dir: str,
    annotations_dir: str,
    epochs: int,
    batch: int,
    device: str,
    skip_hypotheses: bool,
    output_dir: str
) -> None:
    """
    Запускает полный пайплайн лабораторной работы (задание на тройку).

    Все четыре шага используют исключительно ultralytics / YOLOv11.
    Шаг 4 ("самостоятельная имплементация") реализован через обучение
    с нуля (pretrained=False), что изолирует вклад transfer learning.

    Args:
        images_dir: Путь к директории с изображениями.
        annotations_dir: Путь к директории с XML-аннотациями.
        epochs: Число эпох обучения YOLO-моделей (шаги 2-3).
        batch: Размер батча.
        device: Устройство ('cpu' или '0' для GPU).
        skip_hypotheses: Если True — пропустить отдельные гипотезы.
        output_dir: Корневая директория проекта.
    """
    dataset_dir = Path(output_dir) / "dataset"
    configs_dir = Path(output_dir) / "configs"
    results_dir = Path(output_dir) / "results"
    report_dir = Path(output_dir) / "report"

    for d in [configs_dir, results_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    yaml_path = str(configs_dir / "car_plate.yaml")

    # ──────────────────────────────────────────
    # ШАГ 1: Подготовка датасета
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ШАГ 1: Подготовка датасета")
    print("=" * 60)
    result = prepare_dataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=str(dataset_dir)
    )
    generate_yaml(str(dataset_dir), result["class_map"], yaml_path)

    # ──────────────────────────────────────────
    # ШАГ 2: Baseline — YOLOv11n (pretrained)
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ШАГ 2: Baseline YOLOv11n (pretrained=True)")
    print("=" * 60)
    baseline_metrics = train_baseline(
        yaml_path=yaml_path,
        epochs=epochs,
        batch=batch,
        project=str(results_dir),
        device=device
    )
    baseline_metrics["experiment"] = "YOLOv11n baseline (pretrained)"

    # ──────────────────────────────────────────
    # ШАГ 3: Улучшенный baseline (гипотезы)
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ШАГ 3: Проверка гипотез + улучшенный baseline")
    print("=" * 60)

    all_pretrained_metrics = [baseline_metrics]

    if not skip_hypotheses:
        hyp_results = run_hypothesis_experiments(
            yaml_path=yaml_path,
            base_epochs=epochs,
            project=str(results_dir),
            device=device
        )
        all_pretrained_metrics.extend(hyp_results)

    improved_metrics = train_best_config(
        yaml_path=yaml_path,
        epochs=epochs,
        project=str(results_dir),
        device=device
    )
    improved_metrics["experiment"] = "YOLOv11s improved (pretrained)"
    all_pretrained_metrics.append(improved_metrics)

    compare_experiments(all_pretrained_metrics, str(results_dir / "yolo_comparison.txt"))

    # ──────────────────────────────────────────
    # ШАГ 4: Имплементация с нуля (pretrained=False)
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ШАГ 4: YOLOv11 обучение с нуля (pretrained=False)")
    print("=" * 60)

    # Scratch-моделям нужно больше эпох для сходимости без предобученных весов
    scratch_epochs = max(epochs, 100)
    scratch_results = []

    for variant in ["yolo11n", "yolo11s"]:
        m = train_from_scratch(
            yaml_path=yaml_path,
            model_variant=variant,
            epochs=scratch_epochs,
            batch=batch,
            project=str(results_dir),
            name=f"custom_scratch_{variant}",
            device=device,
        )
        scratch_results.append(m)

    compare_with_pretrained(
        scratch_metrics=scratch_results,
        pretrained_results_dir=str(results_dir),
        project=str(results_dir)
    )

    # ──────────────────────────────────────────
    # ШАГ 5: Финальный отчёт
    # ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ШАГ 5: Финальное сравнение всех моделей")
    print("=" * 60)

    all_metrics = all_pretrained_metrics + scratch_results
    print_comparison_table(all_metrics)

    report_path = str(report_dir / "final_report.md")
    generate_markdown_report(all_metrics, report_path)

    print("\n" + "=" * 60)
    print("✓ Пайплайн завершён!")
    print(f"  Отчёт: {report_path}")
    print(f"  Веса:  {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Полный пайплайн лабораторной работы: детекция номерных знаков (YOLOv11)"
    )
    parser.add_argument("--images", required=True,
                        help="Путь к директории images/ датасета")
    parser.add_argument("--annotations", required=True,
                        help="Путь к директории annotations/ датасета")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Эпохи для шагов 2-3 (default: 50). Шаг 4 — max(epochs, 100)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0",
                        help="'0' для GPU, 'cpu' для CPU (default: '0')")
    parser.add_argument("--skip-hypotheses", action="store_true",
                        help="Пропустить отдельные эксперименты H1-H4")
    parser.add_argument("--output-dir", default=".",
                        help="Корневая директория проекта (default: .)")
    args = parser.parse_args()

    run_pipeline(
        images_dir=args.images,
        annotations_dir=args.annotations,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        skip_hypotheses=args.skip_hypotheses,
        output_dir=args.output_dir
    )
