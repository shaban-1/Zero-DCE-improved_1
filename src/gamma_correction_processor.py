import os
import cv2
import re
import numpy as np

def gamma_correction(image, gamma=0.4):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_images(input_dir, dark_output_dir, light_output_dir):
    if not os.path.exists(input_dir):
        print(f"Ошибка: Папка {input_dir} не существует.")
        return

    if not os.path.exists(dark_output_dir):
        os.makedirs(dark_output_dir)
    if not os.path.exists(light_output_dir):
        os.makedirs(light_output_dir)

    image_pattern = re.compile(r".+\.(png|jpg|jpeg)$", re.IGNORECASE)

    for filename in os.listdir(input_dir):
        if image_pattern.match(filename):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Ошибка: Не удалось прочитать изображение {img_path}")
                continue

            for gamma in np.arange(0.4, 1.7, 0.1):
                gamma = round(gamma, 1)
                corrected_image = gamma_correction(image, gamma)

                if corrected_image is None:
                    print(f"Ошибка: Не удалось применить гамма-коррекцию для {filename} с gamma={gamma}")
                    continue

                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_{gamma}{ext}"

                if gamma <= 0.9:
                    output_path = os.path.join(dark_output_dir, new_filename)
                elif gamma >= 1.1:
                    output_path = os.path.join(light_output_dir, new_filename)
                else:
                    continue

                if not os.path.exists(output_path):
                    if not cv2.imwrite(output_path, corrected_image):
                        print(f"Ошибка: Не удалось сохранить изображение {output_path}")
                else:
                    print(f"Файл уже существует: {output_path}")


input_folder = r"./data/test_data/normal"
dark_images_folder = r"./data/test_data/dark"
light_images_folder = r"./data/test_data/light"
process_images(input_folder, dark_images_folder, light_images_folder)