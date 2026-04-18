"""
Скрипт подготовки датасета Car Plate Detection.

Конвертирует аннотации Pascal VOC (XML) в формат YOLO (TXT),
разбивает датасет на обучающую, валидационную и тестовую выборки.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import argparse


def parse_xml_annotation(xml_path: str) -> dict:
    """
    Парсит XML-файл аннотации в формате Pascal VOC.

    Args:
        xml_path: Путь к XML-файлу аннотации.

    Returns:
        Словарь с именем файла, размерами изображения и списком объектов.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return {"filename": filename, "width": width, "height": height, "objects": objects}


def voc_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int,
                img_w: int, img_h: int) -> tuple:
    """
    Конвертирует координаты bounding box из формата Pascal VOC в формат YOLO.

    Args:
        xmin: Левая граница рамки.
        ymin: Верхняя граница рамки.
        xmax: Правая граница рамки.
        ymax: Нижняя граница рамки.
        img_w: Ширина изображения.
        img_h: Высота изображения.

    Returns:
        Кортеж (x_center, y_center, width, height) в нормализованном виде [0..1].
    """
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def build_class_map(annotations_dir: str) -> dict:
    """
    Строит маппинг имён классов → индексы по всем XML-файлам.

    Args:
        annotations_dir: Путь к директории с XML-аннотациями.

    Returns:
        Словарь {class_name: class_index}.
    """
    class_names = set()
    for xml_file in Path(annotations_dir).glob("*.xml"):
        data = parse_xml_annotation(str(xml_file))
        for obj in data["objects"]:
            class_names.add(obj["name"])
    return {name: idx for idx, name in enumerate(sorted(class_names))}


def prepare_dataset(
    images_dir: str,
    annotations_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
) -> dict:
    """
    Подготавливает датасет: конвертирует аннотации и разбивает на сплиты.

    Args:
        images_dir: Путь к директории с PNG-изображениями.
        annotations_dir: Путь к директории с XML-аннотациями.
        output_dir: Путь к выходной директории.
        train_ratio: Доля обучающей выборки.
        val_ratio: Доля валидационной выборки.
        seed: Зерно генератора случайных чисел для воспроизводимости.

    Returns:
        Словарь с количеством примеров в каждом сплите и маппингом классов.
    """
    random.seed(seed)

    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    class_map = build_class_map(annotations_dir)
    print(f"Классы: {class_map}")

    xml_files = sorted(Path(annotations_dir).glob("*.xml"))
    random.shuffle(xml_files)

    n = len(xml_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": xml_files[:n_train],
        "val": xml_files[n_train:n_train + n_val],
        "test": xml_files[n_train + n_val:]
    }

    counts = {}
    for split_name, files in splits.items():
        count = 0
        for xml_file in files:
            data = parse_xml_annotation(str(xml_file))
            img_name = data["filename"]
            img_path = Path(images_dir) / img_name

            # Пробуем найти изображение с учётом расширения
            if not img_path.exists():
                stem = Path(img_name).stem
                found = list(Path(images_dir).glob(f"{stem}.*"))
                if not found:
                    print(f"  [WARN] Изображение не найдено: {img_name}")
                    continue
                img_path = found[0]

            # Копируем изображение
            dst_img = output_path / split_name / "images" / img_path.name
            shutil.copy2(str(img_path), str(dst_img))

            # Записываем YOLO-аннотацию
            dst_label = output_path / split_name / "labels" / (img_path.stem + ".txt")
            with open(dst_label, "w") as f:
                for obj in data["objects"]:
                    cls_idx = class_map[obj["name"]]
                    xc, yc, w, h = voc_to_yolo(
                        obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                        data["width"], data["height"]
                    )
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            count += 1
        counts[split_name] = count
        print(f"  {split_name}: {count} изображений")

    # Сохраняем имена классов
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w") as f:
        for name in sorted(class_map, key=class_map.get):
            f.write(name + "\n")

    return {"counts": counts, "class_map": class_map}


def generate_yaml(output_dir: str, class_map: dict, yaml_path: str) -> None:
    """
    Генерирует YAML-конфиг датасета для ultralytics YOLO.

    Args:
        output_dir: Абсолютный путь к директории датасета.
        class_map: Словарь {class_name: class_index}.
        yaml_path: Путь для сохранения YAML-файла.
    """
    abs_path = str(Path(output_dir).resolve())
    names = {v: k for k, v in class_map.items()}
    names_str = "\n".join(f"  {idx}: {name}" for idx, name in sorted(names.items()))

    yaml_content = f"""# Car Plate Detection Dataset
path: {abs_path}
train: train/images
val: val/images
test: test/images

nc: {len(class_map)}
names:
{names_str}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"YAML-конфиг сохранён: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка датасета Car Plate Detection")
    parser.add_argument("--images", required=True, help="Путь к директории images")
    parser.add_argument("--annotations", required=True, help="Путь к директории annotations")
    parser.add_argument("--output", default="dataset", help="Путь к выходной директории")
    parser.add_argument("--yaml", default="configs/car_plate.yaml", help="Путь к YAML-конфигу")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("=== Подготовка датасета ===")
    result = prepare_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    generate_yaml(args.output, result["class_map"], args.yaml)
    print(f"\nИтого: {result['counts']}")
    print("Датасет готов!")
