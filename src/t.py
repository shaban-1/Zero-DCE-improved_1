import os
from IPython.display import display, HTML


def generate_html_report(results, results_dir="/content/Zero-DCE-improved_1/src/results",
                         report_filename="report.html"):
    """
    Генерирует HTML-отчёт по результатам обработки.

    Параметры:
      results: список словарей с ключами:
               - "filename": имя исходного файла (например, "malignant (98).png")
               - "clahe_metrics": кортеж метрик для CLAHE (mse, psnr, ssim, entropy, edge_intensity, brisque)
               - "zero_dce_metrics": кортеж метрик для Zero-DCE (mse, psnr, ssim, entropy, edge_intensity, brisque)
      results_dir: папка, где находятся сохранённые изображения
      report_filename: имя выходного HTML-файла отчёта
    """
    # Построим HTML-отчёт
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Отчёт по обработке изображений</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 50px; }
        .images { display: flex; gap: 20px; }
        .image-block { text-align: center; }
        .image-block img { max-width: 300px; border: 1px solid #ccc; }
        .metrics { margin-top: 10px; font-family: monospace; font-size: 14px; }
        hr { border: none; border-top: 1px solid #aaa; margin: 40px 0; }
    </style>
</head>
<body>
    <h1>Отчёт по обработке изображений</h1>
"""

    # Используем enumerate для вставки индекса в имена файлов
    for i, res in enumerate(results):
        if(res == results[-1]):
            # Файлы именуются по схеме:
            # original_{i}_{filename}, clahe_{i}_{filename}, zero_dce_{i}_{filename}
            m1 = res["clahe_metrics"]

            m = res["zero_dce_metrics"]

            html += ("<div class='metrics'>"
                     f"<strong>CLAHE</strong> - MSE: {m1[0]:.4f}, PSNR: {m1[1]:.4f}, SSIM: {m1[2]:.4f}, "
                     f"Entropy: {m1[3]:.4f}, Edge Intensity: {m1[4]:.4f}, BRISQUE: {m1[5]:.4f}"
                     "</div>\n")

            html += ("<div class='metrics'>"
                     f"<strong>Zero-DCE</strong> - MSE: {m[0]:.4f}, PSNR: {m[1]:.4f}, SSIM: {m[2]:.4f}, "
                     f"Entropy: {m[3]:.4f}, Edge Intensity: {m[4]:.4f}, BRISQUE: {m[5]:.4f}"
                     "</div>\n")

            html += "<hr>\n"
            break

        # Файлы именуются по схеме:
        # original_{i}_{filename}, clahe_{i}_{filename}, zero_dce_{i}_{filename}
        orig_file = f"original_{i}_{res['filename']}"
        clahe_file = f"clahe_{i}_{res['filename']}"
        zero_file = f"zero_dce_{i}_{res['filename']}"

        html += f"<div class='section'>\n"
        html += f"<h2>Изображение {res['filename']}</h2>\n"
        html += "<div class='images'>\n"

        # Блок для оригинала
        html += "<div class='image-block'>\n"
        html += "<h3>Оригинал</h3>\n"
        html += f"<img src='{orig_file}' alt='Оригинал'>\n"
        html += "</div>\n"

        # Блок для CLAHE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>CLAHE</h3>\n"
        html += f"<img src='{clahe_file}' alt='CLAHE'>\n"
        m1 = res["clahe_metrics"]

        html += "</div>\n"

        # Блок для Zero-DCE с метриками
        html += "<div class='image-block'>\n"
        html += "<h3>Zero-DCE</h3>\n"
        html += f"<img src='{zero_file}' alt='Zero-DCE'>\n"
        m = res["zero_dce_metrics"]

        html += "</div>\n"


        html += "</div>\n"
        html += ("<div class='metrics'>"
                 f"<strong>CLAHE</strong> - MSE: {m1[0]:.4f}, PSNR: {m1[1]:.4f}, SSIM: {m1[2]:.4f}, "
                 f"Entropy: {m1[3]:.4f}, Edge Intensity: {m1[4]:.4f}, BRISQUE: {m1[5]:.4f}"
                 "</div>\n")

        html += ("<div class='metrics'>"
                 f"<strong>Zero-DCE</strong> - MSE: {m[0]:.4f}, PSNR: {m[1]:.4f}, SSIM: {m[2]:.4f}, "
                 f"Entropy: {m[3]:.4f}, Edge Intensity: {m[4]:.4f}, BRISQUE: {m[5]:.4f}"
                 "</div>\n")

        html += "<hr>\n"
        html += "</div>\n"

    html += """
</body>
</html>
"""
    # Сохраняем отчёт в файл
    os.makedirs(results_dir, exist_ok=True)  # <== Добавляем эту строку
    report_path = os.path.join(results_dir, report_filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Отчёт сохранён по пути: {report_path}")

    # Отобразим отчёт в блокноте (если используется Jupyter/Colab)
    display(HTML(html))


# Пример использования
results = [
    {
        "filename": "malignant (98).png",
        "clahe_metrics": (0.0148, 18.2948, 0.7083, 0.0298, 0.0918, 16.7280),
        "zero_dce_metrics": (0.0138, 18.5979, 0.7510, 0.0405, 0.0626, 14.9931)
    },
    {
        "filename": "malignant (54).png",
        "clahe_metrics": (0.0118, 19.2794, 0.7407, 0.0029, 0.0827, 23.5276),
        "zero_dce_metrics": (0.0180, 17.4540, 0.8312, 0.0000, 0.0472, 24.6417)
    },
    {
        "filename": "malignant (36).png",
        "clahe_metrics": (0.0106, 19.7284, 0.7157, 0.0027, 0.0538, 21.9809),
        "zero_dce_metrics": (0.0247, 16.0652, 0.6702, 0.0067, 0.0344, 19.7068)
    },
    {
        "filename": "benign (169).png",
        "clahe_metrics": (0.0138, 18.6150, 0.7419, 0.0067, 0.0877, 34.4830),
        "zero_dce_metrics": (0.0197, 17.0600, 0.8102, 0.0079, 0.0520, 33.2175)
    },
    {
        "filename": "normal (28).png",
        "clahe_metrics": (0.0171, 17.6594, 0.6730, 0.0029, 0.0826, 27.4330),
        "zero_dce_metrics": (0.0183, 17.3720, 0.7037, 0.0005, 0.0591, 24.9747)
    },
    {
        "filename": "malignant (20).png",
        "clahe_metrics": (0.0145, 18.3911, 0.6972, 0.0025, 0.0865, 27.4063),
        "zero_dce_metrics": (0.0190, 17.2204, 0.7595, 0.0005, 0.0544, 27.5195)
    },
    {
        "filename": "benign (27).png",
        "clahe_metrics": (0.0125, 19.0424, 0.7940, 0.0071, 0.1033, 38.3614),
        "zero_dce_metrics": (0.0121, 19.1643, 0.9112, 0.0069, 0.0576, 39.8615)
    },
    {
        "filename": "benign (19).png",
        "clahe_metrics": (0.0184, 17.3509, 0.7509, 0.0020, 0.0928, 34.3295),
        "zero_dce_metrics": (0.0256, 15.9209, 0.8070, 0.0014, 0.0561, 36.2156)
    },
    {
        "filename": "benign (87).png",
        "clahe_metrics": (0.0206, 16.8600, 0.7289, 0.0018, 0.0917, 30.2726),
        "zero_dce_metrics": (0.0207, 16.8485, 0.7940, 0.0003, 0.0586, 29.4695)
    },
    {
        "filename": "malignant (69).png",
        "clahe_metrics": (0.0138, 18.5914, 0.6345, 0.0003, 0.0683, 23.9371),
        "zero_dce_metrics": (0.0177, 17.5194, 0.6616, 0.0000, 0.0492, 24.4052)
    },
    {
        "filename": "Итоговые результаты (средние значения)",
        "clahe_metrics": (0.0148, 18.3813, 0.7185, 0.0059, 0.0841, 27.8459),
        "zero_dce_metrics": (0.0190, 17.3223, 0.7700, 0.0065, 0.0531, 27.5005)
    }
]

# Генерация отчёта (HTML-файл будет сохранён в папке results)
generate_html_report(results, results_dir=r"./results", report_filename="report.html")
