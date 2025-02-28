import os

def cleanup_images(dark_output_dir, light_output_dir):
    """
    Удаляет все файлы в указанных папках.
    """
    for folder in [dark_output_dir, light_output_dir]:
        if os.path.exists(folder):  # Проверяем, существует ли папка
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    os.remove(file_path)
                    print(f"Удалено: {file_path}")
                except Exception as e:
                    print(f"Ошибка при удалении {file_path}: {e}")
        else:
            print(f"Папка {folder} не существует, пропуск.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Очистка временных изображений")
    parser.add_argument("--dark_output_dir", type=str, default="./data/test_data/dark", help="Путь к папке с затемненными изображениями")
    parser.add_argument("--light_output_dir", type=str, default="./data/test_data/light", help="Путь к папке с осветленными изображениями")

    args = parser.parse_args()
    cleanup_images(args.dark_output_dir, args.light_output_dir)