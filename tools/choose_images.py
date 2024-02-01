import os
import random
import shutil
import threading
from tqdm import tqdm

def copy_image(source_path, destination_path):
    shutil.copy(source_path, destination_path)

def random_copy_images(source_folder, destination_folder, num_images, num_threads):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    image_files = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 分割图片列表，为每个线程分配任务
    chunk_size = len(selected_images) // num_threads
    chunks = [selected_images[i:i + chunk_size] for i in range(0, len(selected_images), chunk_size)]

    # 创建并启动线程
    threads = []
    progress_bars = [tqdm(total=len(chunk), position=i, desc=f"Thread {i+1}") for i, chunk in enumerate(chunks)]
    for i, chunk in enumerate(chunks):
        thread = threading.Thread(target=copy_images_thread, args=(chunk, source_folder, destination_folder, progress_bars[i]))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def copy_images_thread(image_files, source_folder, destination_folder, progress_bar):
    for image_file in image_files:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        copy_image(source_path, destination_path)
        progress_bar.update(1)

if __name__ == "__main__":

    """
    1. 该脚本主要是用于 PTQ 量化时校准图片的选取
    2. 主要功能是随机选取训练集中的 N 张图片保存到 workspace/calib_data 文件夹中用于校准
    """

    source_folder = '/home/jarvis/Learn/Datasets/VOC_PTQ/images/train'  # 源文件夹路径
    destination_folder = 'workspace/calib_data'  # 目标文件夹路径
    num_images = 500  # 需要随机获取的图片数量
    num_threads = 1  # 线程数量

    random_copy_images(source_folder, destination_folder, num_images, num_threads)