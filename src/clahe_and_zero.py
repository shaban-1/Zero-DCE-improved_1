import os
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import enhance_net_nopool

# Функция для загрузки модели Zero-DCE
def load_model():
    model = enhance_net_nopool().cuda()
    model.load_state_dict(torch.load("snapshots/Epoch100.pth"))
    model.eval()
    return model

# Функция для улучшения изображения с помощью Zero-DCE (grayscale)
def enhance_with_zero_dce(model, image_path):
    img = PILImage.open(image_path).convert('L')  # Преобразуем в grayscale
    img = img.resize((256, 256), PILImage.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        enhanced_image, _, _ = model(img)  # Выход модели (1, 3, 256, 256)

    # Преобразуем изображение в grayscale
    enhanced_image = enhanced_image.squeeze().cpu().numpy()  # (3, 256, 256)
    if enhanced_image.ndim == 3:
        enhanced_image = np.mean(enhanced_image, axis=0)  # Преобразуем RGB -> Grayscale

    enhanced_image = np.clip(enhanced_image, 0, 1)
    return enhanced_image

# Функция для вычисления метрик
def calculate_metrics(original, enhanced):
    assert original.shape == enhanced.shape, f"Shapes mismatch: {original.shape} vs {enhanced.shape}"

    mse = np.mean((original - enhanced) ** 2)
    psnr_val = psnr(original, enhanced, data_range=1.0)
    ssim_val = ssim(original, enhanced, data_range=1.0)
    return mse, psnr_val, ssim_val

# Путь к директории с тестовыми изображениями
input_dir = "C:/Users/sevda/PycharmProjects/Neural Network/Zero-DCE-improved/src/data/test_data/DICM/"

# Загрузка модели Zero-DCE
model = load_model()

# Списки для хранения метрик
clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []

# Переменные для хранения лучших изображений
best_clahe_ssim = -1
best_clahe_image = None
best_clahe_filename = ""

best_zero_dce_ssim = -1
best_zero_dce_image = None
best_zero_dce_filename = ""

# Обработка всех изображений в директории
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in filename:
        image_path = os.path.join(input_dir, filename)

        # Загрузка и предобработка изображения
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Изменение размера до 256x256
        original_resized = cv2.resize(image, (256, 256))
        original_float = original_resized.astype(np.float32) / 255.0

        # Применение CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(original_resized)
        clahe_float = clahe_image.astype(np.float32) / 255.0

        # Применение Zero-DCE
        zero_dce_image = enhance_with_zero_dce(model, image_path)

        # Вычисление метрик для CLAHE
        clahe_mse, clahe_psnr, clahe_ssim_val = calculate_metrics(original_float, clahe_float)
        clahe_mse_list.append(clahe_mse)
        clahe_psnr_list.append(clahe_psnr)
        clahe_ssim_list.append(clahe_ssim_val)

        # Проверка на лучшее изображение CLAHE
        if clahe_ssim_val > best_clahe_ssim:
            best_clahe_ssim = clahe_ssim_val
            best_clahe_image = clahe_image
            best_clahe_filename = filename

        # Вычисление метрик для Zero-DCE
        zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val = calculate_metrics(original_float, zero_dce_image)
        zero_dce_mse_list.append(zero_dce_mse)
        zero_dce_psnr_list.append(zero_dce_psnr)
        zero_dce_ssim_list.append(zero_dce_ssim_val)

        # Проверка на лучшее изображение Zero-DCE
        if zero_dce_ssim_val > best_zero_dce_ssim:
            best_zero_dce_ssim = zero_dce_ssim_val
            best_zero_dce_image = (zero_dce_image * 255).astype(np.uint8)
            best_zero_dce_filename = filename

        # Вывод результатов для текущего изображения
        print(f"Результаты для {filename}:")
        print(f"  CLAHE Metrics:")
        print(f"    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}")
        print(f"  Zero-DCE Metrics:")
        print(f"    MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}")
        print("-" * 50)

# Сохранение лучших изображений
if best_clahe_image is not None:
    cv2.imwrite("best_clahe_image.png", best_clahe_image)
    print(f"Лучшее изображение CLAHE сохранено как 'best_clahe_image.png' (Файл: {best_clahe_filename})")

if best_zero_dce_image is not None:
    cv2.imwrite("best_zero_dce_image.png", best_zero_dce_image)
    print(f"Лучшее изображение Zero-DCE сохранено как 'best_zero_dce_image.png' (Файл: {best_zero_dce_filename})")

# Вывод итоговых результатов
if len(clahe_mse_list) > 0:
    print("Итоговые результаты (средние значения):")
    print(f"  CLAHE Metrics:")
    print(f"    MSE: {np.mean(clahe_mse_list):.4f}, PSNR: {np.mean(clahe_psnr_list):.4f}, SSIM: {np.mean(clahe_ssim_list):.4f}")
    print(f"  Zero-DCE Metrics:")
    print(f"    MSE: {np.mean(zero_dce_mse_list):.4f}, PSNR: {np.mean(zero_dce_psnr_list):.4f}, SSIM: {np.mean(zero_dce_ssim_list):.4f}")
else:
    print("Нет изображений для обработки.")
