import os
import cv2
import re
import numpy as np


# Функция для гамма-коррекции
def gamma_correction(image, gamma=0.4):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


# Функция для обработки изображений
def process_images(input_dir, dark_output_dir, light_output_dir):
    if not os.path.exists(dark_output_dir):
        os.makedirs(dark_output_dir)
    if not os.path.exists(light_output_dir):
        os.makedirs(light_output_dir)

    benign_pattern = re.compile(r"^benign \(\d+\)\.(png|jpg|jpeg)$", re.IGNORECASE)

    for filename in os.listdir(input_dir):
        if benign_pattern.match(filename):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue

            for gamma in np.arange(0.4, 1.7, 0.1):
                gamma = round(gamma, 1)
                corrected_image = gamma_correction(image, gamma)

                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{gamma}{ext}"

                if gamma <= 0.9:
                    output_path = os.path.join(dark_output_dir, new_filename)
                elif gamma >= 1.1:
                    output_path = os.path.join(light_output_dir, new_filename)
                else:
                    continue  # Пропускаем гамму 1.0

                # Проверяем, существует ли файл
                if not os.path.exists(output_path):
                    cv2.imwrite(output_path, corrected_image)
                else:
                    print(f"Файл уже существует: {output_path}")

# Пути к папкам
input_folder = r"./data/test_data/normal"
dark_images_folder = r"./data/test_data/dark"
light_images_folder = r"./data/test_data/light"

# Обработка изображений
process_images(input_folder, dark_images_folder, light_images_folder)