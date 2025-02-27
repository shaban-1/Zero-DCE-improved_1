import os
import cv2
import re
import numpy as np

def gamma_correction(image, gamma=1.0):
    """Применяет гамма-коррекцию для затемнения изображения"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_images(input_dir, output_dir, gamma=1.0):
    """Читает изображения, затемняет их и сохраняет в новую папку с приписыванием "1" к имени"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Регулярное выражение для файлов "benign (число).png"
    benign_pattern = re.compile(r"^benign \(\d+\)\.(png|jpg|jpeg)$", re.IGNORECASE)

    for filename in os.listdir(input_dir):
        if benign_pattern.match(filename):  # Проверяем, соответствует ли имя маске
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue

            dark_image = gamma_correction(image, gamma)

            # Разделяем имя и расширение
            name, ext = os.path.splitext(filename)  # Например, "benign (12)", ".png"
            new_filename = f"{name}{ext}"  # Добавляем "1" перед расширением
            output_path = os.path.join(output_dir, new_filename)

            cv2.imwrite(output_path, dark_image)
            print(f"Затемнено и сохранено: {new_filename}")

# Папки
input_folder = r"./data/train_data"
output_folder = r"./data/test_data/DICM"

# Запуск
process_images(input_folder, output_folder)
