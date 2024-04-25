import cv2
import numpy as np
import os


def segment_objects(original_image, mask_folder, output_folder):
    segmented_images = []

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历掩码图像文件夹中的每个文件
    for filename in os.listdir(mask_folder):
        # 确保文件是图像文件
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取掩码图像
            mask_image = cv2.imread(os.path.join(mask_folder, filename))

            # 将掩码图转换为二值图像
            mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # 获取对象实例的轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 创建白色背景图
            x, y, w, h = cv2.boundingRect(contours[0])
            background = np.ones((h, w, 3), dtype=np.uint8) * 255

            # 将原始图像按 mask 进行像素级别的截取，并放置在白色背景上
            original_roi = original_image[y:y + h, x:x + w]
            mask_roi = mask[y:y + h, x:x + w]
            background[mask_roi > 0] = original_roi[mask_roi > 0]

            # 生成输出文件路径
            output_path = os.path.join(output_folder, f'segmented_{filename}')

            # 保存分割后的图像
            cv2.imwrite(output_path, background)

            segmented_images.append(background)

    return segmented_images


# 加载原始图像
original_image = cv2.imread(r"C:\project download\talior\carr.jpg")

# 指定存储掩码图像的文件夹路径
mask_folder = r"C:\project download\talior\car"

# 指定存储分割后图像的文件夹路径
output_folder = r"C:\project download\talior\resault\wbtest5"

# 分割对象实例并保存图像
segmented_images = segment_objects(original_image, mask_folder, output_folder)

print("分割后的图像已保存到:", output_folder)
