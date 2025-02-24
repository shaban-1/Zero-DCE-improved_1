import os
import argparse
import cv2
import numpy as np
from PIL import Image as PILImage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import enhance_net_nopool
from skimage.filters import sobel
import torch
import piq


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
    img = PILImage.open(image_path).convert('L')
    img = img.resize((256, 256), PILImage.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        enhanced_image, _, _ = model(img)

    enhanced_image = enhanced_image.squeeze().cpu().numpy()
    if enhanced_image.ndim == 3:
        enhanced_image = np.mean(enhanced_image, axis=0)  # RGB -> Grayscale

    enhanced_image = np.clip(enhanced_image, 0, 1)

    enhanced_image_uint8 = (enhanced_image * 255).astype(np.uint8)
    gamma_corrected = gamma_trans(enhanced_image_uint8, 1.2)
    gamma_corrected = gamma_corrected.astype(np.float32) / 255.0

    return gamma_corrected


def calculate_brisque(image):
    if isinstance(image, np.ndarray):
        image = np.clip(image, 0, 1)
    else:
        raise TypeError("Expected image to be a numpy array")

    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    return piq.brisque(image_tensor)


# ЭНТРОПИЯ
def calculate_entropy(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum([p * np.log2(p) for p in histogram if p != 0])
    return entropy

# РЕЗКОСТЬ КРАЁВ
def calculate_edge_intensity(image):
    edges = sobel(image)
    return np.mean(edges)


def calculate_metrics(original, enhanced):
    assert original.shape == enhanced.shape, f"Shapes mismatch: {original.shape} vs {enhanced.shape}"

    mse = np.mean((original - enhanced) ** 2)
    psnr_val = psnr(original, enhanced, data_range=1.0)
    ssim_val = ssim(original, enhanced, data_range=1.0)
    entropy_val = calculate_entropy(enhanced)
    edge_intensity = calculate_edge_intensity(enhanced)
    brisque_val = calculate_brisque(enhanced)

    return mse, psnr_val, ssim_val, entropy_val, edge_intensity, brisque_val


# Обработка изображений
def process_images(input_dir, model):
    # Списки для всех метрик
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list = [], [], []
    clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list = [], [], []

    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list = [], [], []
    zero_dce_entropy_list, zero_dce_edge_intensity_list, zero_dce_brisque_list = [], [], []

    best_combined_ssim = -1
    best_image_data = None

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

            # Метрики для CLAHE
            clahe_mse, clahe_psnr, clahe_ssim_val, clahe_entropy, clahe_edge_intensity, clahe_brisque = calculate_metrics(
                original_float, clahe_float)
            clahe_mse_list.append(clahe_mse)
            clahe_psnr_list.append(clahe_psnr)
            clahe_ssim_list.append(clahe_ssim_val)
            clahe_entropy_list.append(clahe_entropy)
            clahe_edge_intensity_list.append(clahe_edge_intensity)
            clahe_brisque_list.append(clahe_brisque)

            # Метрики для Zero-DCE
            zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val, zero_dce_entropy, zero_dce_edge_intensity, zero_dce_brisque = calculate_metrics(
                original_float, zero_dce_image)
            zero_dce_mse_list.append(zero_dce_mse)
            zero_dce_psnr_list.append(zero_dce_psnr)
            zero_dce_ssim_list.append(zero_dce_ssim_val)
            zero_dce_entropy_list.append(zero_dce_entropy)
            zero_dce_edge_intensity_list.append(zero_dce_edge_intensity)
            zero_dce_brisque_list.append(zero_dce_brisque)

            # Выбор лучшего изображения по среднему SSIM
            # Разница между метриками
            delta_ssim = zero_dce_ssim_val - clahe_ssim_val
            delta_psnr = zero_dce_psnr - clahe_psnr
            delta_entropy = zero_dce_entropy - clahe_entropy
            delta_edge_intensity = zero_dce_edge_intensity - clahe_edge_intensity
            delta_brisque = clahe_brisque - zero_dce_brisque  # Чем меньше, тем лучше

            # Вычисляем итоговый скор
            score = (2.0 * delta_ssim) + (1.5 * delta_psnr) + (1.0 * delta_entropy) + (1.0 * delta_edge_intensity) + (
                        1.5 * delta_brisque)

            # Выбор лучшего изображения по наибольшей разнице в пользу Zero-DCE
            if (
                    delta_ssim > 0 and delta_psnr > 0 and delta_entropy > 0 and delta_edge_intensity > 0 and delta_brisque > 0) and (
                    score > best_combined_ssim):
                best_combined_ssim = score
                best_image_data = {
                    "filename": filename,
                    "original_image": original_float,
                    "clahe_image": clahe_image,
                    "zero_dce_image": (zero_dce_image * 255).astype(np.uint8)
                }

            # Вывод метрик для каждого изображения
            print(f"Результаты для {filename}:")
            print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}, "
                  f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge_intensity:.4f}, BRISQUE: {clahe_brisque:.4f}")
            print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}, "
                  f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge_intensity:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
            print("-" * 50)

    return (
    clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
    zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list,
    zero_dce_brisque_list,
    best_image_data)


# Сохранение лучших изображений с выводом метрик
def save_best_images(best_image_data, output_dir="best_images"):
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    if best_image_data is not None:
        filename = best_image_data["filename"]
        original_image = best_image_data["original_image"]
        clahe_image = best_image_data["clahe_image"]
        zero_dce_image = best_image_data["zero_dce_image"]

        # Сохраняем изображения
        cv2.imwrite(os.path.join(output_dir, "best_clahe_image.png"), clahe_image)
        cv2.imwrite(os.path.join(output_dir, "best_zero_dce_image.png"), zero_dce_image)
        print(f"Сохранено лучшее изображение ({filename}) как best_clahe_image.png и best_zero_dce_image.png")

        # Метрики для CLAHE
        clahe_mse, clahe_psnr, clahe_ssim_val, clahe_entropy, clahe_edge_intensity, clahe_brisque = calculate_metrics(
            original_image, clahe_image / 255.0)  # Сравнение с оригиналом
        print(f"  CLAHE -    MSE: {clahe_mse:.4f}, PSNR: {clahe_psnr:.4f}, SSIM: {clahe_ssim_val:.4f}, "
              f"Entropy: {clahe_entropy:.4f}, Edge Intensity: {clahe_edge_intensity:.4f}, BRISQUE: {clahe_brisque:.4f}")

        # Метрики для Zero-DCE
        zero_dce_mse, zero_dce_psnr, zero_dce_ssim_val, zero_dce_entropy, zero_dce_edge_intensity, zero_dce_brisque = calculate_metrics(
            original_image, zero_dce_image / 255.0)
        print(f"  Zero-DCE - MSE: {zero_dce_mse:.4f}, PSNR: {zero_dce_psnr:.4f}, SSIM: {zero_dce_ssim_val:.4f}, "
              f"Entropy: {zero_dce_entropy:.4f}, Edge Intensity: {zero_dce_edge_intensity:.4f}, BRISQUE: {zero_dce_brisque:.4f}")
        print("-" * 50)



# Основная функция
def main(input_dir, output_dir, model_path):
    model = load_model(model_path)

    (clahe_mse_list, clahe_psnr_list, clahe_ssim_list, clahe_entropy_list, clahe_edge_intensity_list, clahe_brisque_list,
     zero_dce_mse_list, zero_dce_psnr_list, zero_dce_ssim_list, zero_dce_entropy_list, zero_dce_edge_intensity_list, zero_dce_brisque_list,
     best_image_data) = process_images(input_dir, model)

    save_best_images(best_image_data, output_dir)

    if len(clahe_mse_list) > 0:
        clahe_avg_mse = np.mean(clahe_mse_list)
        clahe_avg_psnr = np.mean(clahe_psnr_list)
        clahe_avg_ssim = np.mean(clahe_ssim_list)
        clahe_avg_entropy = np.mean(clahe_entropy_list)
        clahe_avg_edge_intensity = np.mean(clahe_edge_intensity_list)
        clahe_avg_brisque = np.mean(clahe_brisque_list)

        zero_dce_avg_mse = np.mean(zero_dce_mse_list)
        zero_dce_avg_psnr = np.mean(zero_dce_psnr_list)
        zero_dce_avg_ssim = np.mean(zero_dce_ssim_list)
        zero_dce_avg_entropy = np.mean(zero_dce_entropy_list)
        zero_dce_avg_edge_intensity = np.mean(zero_dce_edge_intensity_list)
        zero_dce_avg_brisque = np.mean(zero_dce_brisque_list)

        # Вывод итоговых результатов
        print("Итоговые результаты (средние значения):")
        print(f"  CLAHE -    MSE: {clahe_avg_mse:.4f}, PSNR: {clahe_avg_psnr:.4f}, SSIM: {clahe_avg_ssim:.4f}, "
              f"Average Entropy: {clahe_avg_entropy:.4f}, Edge Intensity: {clahe_avg_edge_intensity:.4f}, BRISQUE: {clahe_avg_brisque:.4f}")
        print(f"  Zero-DCE - MSE: {zero_dce_avg_mse:.4f}, PSNR: {zero_dce_avg_psnr:.4f}, SSIM: {zero_dce_avg_ssim:.4f}, "
              f"Average Entropy: {zero_dce_avg_entropy:.4f}, Edge Intensity: {zero_dce_avg_edge_intensity:.4f}, BRISQUE: {zero_dce_avg_brisque:.4f}")
    else:
        print("Нет изображений для обработки.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Enhancement with Zero-DCE and CLAHE + Gamma Correction")
    parser.add_argument("--input_dir", type=str, default="C:/Users/sevda/PycharmProjects/Neural Network/Zero-DCE-improved_1/src/data/test_data/DICM/", help="Путь к директории с изображениями для обработки")
    parser.add_argument("--output_dir", type=str, default="best_images", help="Путь для сохранения лучших изображений")
    parser.add_argument("--model_path", type=str, default="snapshots/Epoch200.pth", help="Путь к модели Zero-DCE")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.model_path)