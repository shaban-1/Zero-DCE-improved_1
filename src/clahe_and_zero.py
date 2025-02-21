import os
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import enhance_net_nopool

# Функция для загрузки модели Zero-DCE
def load_model():
    model = enhance_net_nopool().cuda()
    model.load_state_dict(torch.load("snapshots/Epoch31.pth", weights_only=True))
    model.eval()
    return model

# Функция для улучшения изображения с помощью Zero-DCE (grayscale)
def enhance_with_zero_dce(model, image_path):
    img = PILImage.open(image_path).convert('L')  # Преобразуем в grayscale
    img = img.resize((256, 256), PILImage.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, 256, 256)
    with torch.no_grad():
        enhanced_image, _, _ = model(img)
    enhanced_image = enhanced_image.squeeze().cpu().numpy()  # (256, 256)
    enhanced_image = np.clip(enhanced_image, 0, 1)  # Ограничение значений [0, 1]
    return enhanced_image

# Функция для вычисления метрик (grayscale)
def calculate_metrics(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    psnr_val = psnr(original, enhanced, data_range=1.0)
    ssim_val = ssim(original, enhanced, data_range=1.0, win_size=3)
    return mse, psnr_val, ssim_val

# Путь к директории с тестовыми изображениями
input_dir = "/content/Zero-DCE-improved/src/data/test_data/DICM/"

# Загрузка модели Zero-DCE
model = load_model()

# Списки для хранения метрик
clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []

# Обработка всех изображений в директории
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in filename:  # Пропускаем файлы с маской
        image_path = os.path.join(input_dir, filename)

        # Загрузка исходного изображения в режиме grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Пропущен файл {filename}: не удается прочитать.")
            continue

        original_resized = cv2.resize(image, (256, 256))
        original_float = original_resized.astype(np.float32) / 255.0

        # Применение CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(original_resized)
        clahe_resized = cv2.resize(clahe_image, (256, 256))
        clahe_float = clahe_resized.astype(np.float32) / 255.0

        # Улучшение изображения с помощью Zero-DCE
        zero_dce_image = enhance_with_zero_dce(model, image_path)

        # Выравнивание размеров изображений
        zero_dce_resized = cv2.resize(zero_dce_image, (256, 256))

        # Вычисление метрик для CLAHE
        clahe_mse, clahe_psnr, clahe_ssim = calculate_metrics(original_float, clahe_float)
        clahe_mse_list.append(clahe_mse)
        clahe_psnr_list.append(clahe_psnr)
        clahe_ssim_list.append(clahe_ssim)

        # Вычисление метрик для Zero-DCE
        zero_dce_mse, zero_dce_psnr, zero_dce_ssim = calculate_metrics(original_float, zero_dce_resized)
        zero_dce_mse_list.append(zero_dce_mse)
        zero_dce_psnr_list.append(zero_dce_psnr)
        zero_dce_ssim_list.append(zero_dce_ssim)

        # Вывод результатов для текущего изображения
        print(f"Результаты для {filename}:")
        print(f"  CLAHE Metrics:")
        print(f"    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim:.4f}")
        print(f"  Zero-DCE Metrics:")
        print(f"    MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim:.4f}")
        print("-" * 50)

# Вывод итоговых результатов
if len(clahe_mse_list) > 0:
    print("Итоговые результаты (средние значения):")
    print(f"  CLAHE Metrics:")
    print(f"    MSE: {np.mean(clahe_mse_list):.4f}, PSNR: {np.mean(clahe_psnr_list):.4f}, SSIM: {np.mean(clahe_ssim_list):.4f}")
    print(f"  Zero-DCE Metrics:")
    print(f"    MSE: {np.mean(zero_dce_mse_list):.4f}, PSNR: {np.mean(zero_dce_psnr_list):.4f}, SSIM: {np.mean(zero_dce_ssim_list):.4f}")
else:
    print("Нет изображений для обработки.")