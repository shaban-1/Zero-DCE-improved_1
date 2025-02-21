import os
import argparse
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import enhance_net_nopool

#  Функция гамма-коррекции
def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

# Загрузка модели Zero-DCE
def load_model(model_path):
    model = enhance_net_nopool().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Улучшение с Zero-DCE + гамма-коррекция
def enhance_with_zero_dce(model, image_path):
    img = PILImage.open(image_path).convert('L')  # Преобразуем в grayscale
    img = img.resize((256, 256), PILImage.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        enhanced_image, _, _ = model(img)

    enhanced_image = enhanced_image.squeeze().cpu().numpy()
    if enhanced_image.ndim == 3:
        enhanced_image = np.mean(enhanced_image, axis=0)  # RGB -> Grayscale

    enhanced_image = np.clip(enhanced_image, 0, 1)

    # 🟡 Применяем гамма-коррекцию (γ = 1.15)
    enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)
    gamma_corrected = gamma_trans(enhanced_image_uint8, 1.15)
    gamma_corrected = gamma_corrected.astype(np.float32) / 255.0

    return gamma_corrected

# Вычисление метрик
def calculate_metrics(original, enhanced):
    assert original.shape == enhanced.shape, f"Shapes mismatch: {original.shape} vs {enhanced.shape}"
    mse = np.mean((original - enhanced) ** 2)
    psnr_val = psnr(original, enhanced, data_range=1.0)
    ssim_val = ssim(original, enhanced, data_range=1.0)
    return mse, psnr_val, ssim_val

# Обработка изображений
def process_images(input_dir, model):
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []

    best_clahe_ssim = -1
    best_zero_dce_ssim = -1
    best_clahe_image, best_zero_dce_image = None, None
    best_clahe_filename, best_zero_dce_filename = "", ""

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')) and "_mask" not in filename:
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            original_resized = cv2.resize(image, (256, 256))
            original_float = original_resized.astype(np.float32) / 255.0

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(original_resized)
            clahe_float = clahe_image.astype(np.float32) / 255.0

            # Zero-DCE + Гамма
            zero_dce_image = enhance_with_zero_dce(model, image_path)

            # Метрики
            clahe_mse, clahe_psnr, clahe_ssim_val = calculate_metrics(original_float, clahe_float)
            clahe_mse_list.append(clahe_mse)
            clahe_psnr_list.append(clahe_psnr)
            clahe_ssim_list.append(clahe_ssim_val)

            zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val = calculate_metrics(original_float, zero_dce_image)
            zero_dce_mse_list.append(zero_dce_mse)
            zero_dce_psnr_list.append(zero_dce_psnr)
            zero_dce_ssim_list.append(zero_dce_ssim_val)

            # Лучшая картинка
            if clahe_ssim_val > best_clahe_ssim:
                best_clahe_ssim = clahe_ssim_val
                best_clahe_image = clahe_image
                best_clahe_filename = filename

            if zero_dce_ssim_val > best_zero_dce_ssim:
                best_zero_dce_ssim = zero_dce_ssim_val
                best_zero_dce_image = (zero_dce_image * 255).astype(np.uint8)
                best_zero_dce_filename = filename

            print(f"Результаты для {filename}:")
            print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}")
            print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}")
            print("-" * 50)

    return (clahe_mse_list, clahe_psnr_list, clahe_ssim_list,
            zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list,
            best_clahe_image, best_zero_dce_image, best_clahe_filename, best_zero_dce_filename)

# Сохранение лучших изображений
def save_best_images(best_clahe_image, best_zero_dce_image, best_clahe_filename, best_zero_dce_filename, output_dir="best_images"):
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    if best_clahe_image is not None:
        cv2.imwrite(os.path.join(output_dir, "best_clahe_image.png"), best_clahe_image)
        print(f"Сохранено: {best_clahe_filename} как best_clahe_image.png")

    if best_zero_dce_image is not None:
        cv2.imwrite(os.path.join(output_dir, "best_zero_dce_image.png"), best_zero_dce_image)
        print(f"Сохранено: {best_zero_dce_filename} как best_zero_dce_image.png")

# Основная функция
def main(input_dir, output_dir, model_path):
    model = load_model(model_path)

    (clahe_mse_list, clahe_psnr_list, clahe_ssim_list,
     zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list,
     best_clahe_image, best_zero_dce_image,
     best_clahe_filename, best_zero_dce_filename) = process_images(input_dir, model)

    save_best_images(best_clahe_image, best_zero_dce_image, best_clahe_filename, best_zero_dce_filename, output_dir)

    if len(clahe_mse_list) > 0:
        print("Итоговые результаты (средние значения):")
        print(f"  CLAHE -    MSE: {np.mean(clahe_mse_list):.4f}, PSNR: {np.mean(clahe_psnr_list):.4f}, SSIM: {np.mean(clahe_ssim_list):.4f}")
        print(f"  Zero-DCE - MSE: {np.mean(zero_dce_mse_list):.4f}, PSNR: {np.mean(zero_dce_psnr_list):.4f}, SSIM: {np.mean(zero_dce_ssim_list):.4f}")
    else:
        print("Нет изображений для обработки.")

# Запуск
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement with Zero-DCE and CLAHE + Gamma Correction")
    parser.add_argument("--input_dir", type=str, default="C:/Users/sevda/PycharmProjects/Neural Network/Zero-DCE-improved/src/data/test_data/DICM/", help="Путь к директории с изображениями для обработки")
    parser.add_argument("--output_dir", type=str, default="best_images", help="Путь для сохранения лучших изображений")
    parser.add_argument("--model_path", type=str, default="snapshots/Epoch199.pth", help="Путь к модели Zero-DCE")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_path)
