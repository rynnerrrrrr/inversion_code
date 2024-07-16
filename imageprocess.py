from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt


# 假设你已经有了一个Chl-a反演算法
def chla_inversion(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]
    
    green_div_blue = green / (blue + np.finfo(float).eps)  # 防止除零
    
    # 使用幂函数形式
    a = 6.25
    b = 4.126
    chla_concentration = a*green_div_blue - b
    
    return chla_concentration

#假设建立了一个TSS反演算法
def TSS_inversion(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]
    
    green_div_blue = green / (blue + np.finfo(float).eps)  # 防止除零
    
    # 使用幂函数形式
    a = 6.25
    b = 4.126
    TSS_concentration = a*green_div_blue - b
    
    return TSS_concentration

# 读取上传的影像数据
def read_image(file_path):
    dataset = gdal.Open(file_path)
    image = dataset.ReadAsArray()
    image = np.transpose(image, (1, 2, 0))
    return image

# 生成Chl-a浓度分布图
def generate_chla_map(chla_concentration, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(chla_concentration, cmap='viridis')
    plt.colorbar(label='Chl-a Concentration')
    plt.title('Chl-a Concentration Map')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

# 主函数
def main(file_path):
    # 读取影像数据
    image = read_image(file_path)
    
    # 进行Chl-a反演
    chla_concentration = chla_inversion(image)
    
    # 生成并展示Chl-a浓度分布图
    output_path = 'chla_map.png'
    generate_chla_map(chla_concentration, output_path)
    plt.show()

# 示例调用
if __name__ == "__main__":
    file_path = r"D:\HeDi_HJ2\H2_clipped_heidi.tif"  # 替换为你的影像文件路径
    main(file_path)



