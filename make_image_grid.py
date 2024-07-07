import os
from PIL import Image


def create_image_grid(directory, output_file, images_per_row=10):
    # 获取目录中所有的.jpg文件
    images = [os.path.join(directory, f)
              for f in os.listdir(directory) if f.endswith('.jpg')]

    # 如果图片不足50张，取全部图片
    if len(images) > 50:
        images = images[:50]

    # 读取图片
    images = [Image.open(img) for img in images]

    # 确定每张图片的大小
    img_width, img_height = images[0].size

    # 计算整个网格的宽度和高度
    grid_width = img_width * images_per_row
    grid_height = img_height * \
        (len(images) // images_per_row + (1 if len(images) % images_per_row else 0))

    # 创建一个新的图片用于存放网格
    grid_img = Image.new('RGB', (grid_width, grid_height))

    # 拼接图片
    for index, img in enumerate(images):
        x = index % images_per_row * img_width
        y = index // images_per_row * img_height
        grid_img.paste(img, (x, y))

    # 保存图片
    grid_img.save(output_file)
    print(f"Image grid saved to {output_file}")


# 调用函数
create_image_grid('analysis/result/ResNet1.0/cams', 'output_grid.jpg')
