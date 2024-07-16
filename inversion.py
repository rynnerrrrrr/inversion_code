import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from xgboost import XGBRegressor
import joblib

#加载已经保存好的模型
xgb_chla_model = joblib.load('xgb_chla_model.pkl')
xgb_tss_model = joblib.load('xgb_tss_model.pkl')


def standardize_data(data_array):
    """
    对数据进行标准化处理。
    
    参数:
    data_array (np.ndarray): 输入的二维数据数组。
    
    返回:
    np.ndarray: 标准化后的数据数组。
    """
    mean = np.mean(data_array, axis=0)   #对数据的每一列求平均值，得到包含每一列数据的均值的数组
    std = np.std(data_array, axis=0)     #对数据的每一列求标准差，得到包含每一列数据的标准差的数组
    standardized_data = (data_array - mean) / std     #用到了numpy数组中的广播运算（自动补齐数组，保证两个数组能够正常进行运算）
    return standardized_data

def extract_and_process_bands(image):
    """
    提取图像的各个波段并进行标准化处理。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    
    返回:
    np.ndarray: 标准化后的像素特征数组。
    """
    blue = image[:, :, 0] / 10000    #除以10000是因为影像数据没有进行相应的大气校正
    green = image[:, :, 1] / 10000
    red = image[:, :, 2] / 10000
    nir = image[:, :, 3] / 10000
    
    # 将图像数据重塑为二维数组，每一行代表一个像素的特征向量
    height, width = image.shape[:2]
    pixel_features = np.stack([blue, green, red, nir], axis=-1).reshape(-1, 4)
    standardized_features = standardize_data(pixel_features)
    
    return standardized_features, height, width

def chla_inversion(image, model):
    """
    使用机器学习模型进行叶绿素a浓度的反演。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    model: 预训练的机器学习模型。
    
    返回:
    np.ndarray: 预测的叶绿素a浓度图像。
    """
    # 提取并处理图像波段
    pixel_features, height, width = extract_and_process_bands(image)
    
    # 使用模型进行批量预测
    chla_concentration = model.predict(pixel_features)
    
    # 将预测结果重塑为图像的形状
    chla_concentration = chla_concentration.reshape(height, width)
    
    return chla_concentration


#TSS机器学习反演
def TSS_inversion(image, model):
    """
    使用机器学习模型进行悬浮物浓度的反演。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    model: 预训练的机器学习模型。
    
    返回:
    np.ndarray: 预测的TSS浓度图像。
    """
    # 提取并处理图像波段
    pixel_features, height, width = extract_and_process_bands(image)
    
    # 使用模型进行批量预测
    TSS_concentration = model.predict(pixel_features)
    
    # 将预测结果重塑为图像的形状
    TSS_concentration = TSS_concentration.reshape(height, width)
    
    return TSS_concentration


# 读取影像数据
def read_image(file_path):
    dataset = gdal.Open(file_path)
    image = dataset.ReadAsArray()
    image = np.transpose(image, (1, 2, 0))
    return image

# 生成Chl-a浓度分布图
def generate_chla_map(chla_concentration, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(chla_concentration, cmap='viridis', vmin=np.min(chla_concentration), vmax=np.max(chla_concentration))
    plt.colorbar(label='Chl-a Concentration')
    plt.title('Chl-a Concentration Map')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

#生成chla、TSS的geotiff栅格格式图
def create_geotiff(output_path, width, height, data, geotransform, projection, data_type=gdal.GDT_Float32):
    """
    创建并保存GeoTIFF文件。

    参数:
    output_path (str): 输出文件路径。
    width (int): 影像宽度。
    height (int): 影像高度。
    data (np.ndarray): 要保存的数据。
    geotransform (tuple): 地理变换参数。
    projection (str): 投影信息。
    data_type (int): 数据类型，默认为gdal.GDT_Float32。
    """
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_ds = gtiff_driver.Create(output_path, width, height, 1, data_type)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_ds = None  # 关闭文件

def generate_geotiff_map(file_path, concentration, output_path):
    """
    生成GeoTIFF地图。

    参数:
    file_path (str): 输入影像文件路径。
    concentration (np.ndarray): 浓度数据。
    output_path (str): 输出文件路径。
    """
    try:
        # 获取原影像数据的基本信息
        in_ds = gdal.Open(file_path)
        if in_ds is None:
            raise ValueError(f"无法打开文件: {file_path}")

        in_band = in_ds.GetRasterBand(1)
        geotrans = in_ds.GetGeoTransform()
        proj = in_ds.GetProjection()

        # 创建输出栅格影像
        create_geotiff(output_path, in_band.XSize, in_band.YSize, concentration, geotrans, proj)

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        in_ds = None  # 确保输入文件被关闭

# 示例调用
# generate_geotiff_map('input.tif', concentration_data, 'output.tif')



#生成TSS浓度分布图
def generate_tss_map(tss_concentration, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(tss_concentration, cmap='viridis',vmin=np.min(tss_concentration), vmax=np.max(tss_concentration))
    plt.colorbar(label='TSS Concentration')
    plt.title('TSS Concentration Map')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()



def main():
    file_path = r"D:\HeDi_HJ2\H2_clipped_heidi.tif"
    image = read_image(file_path)
    chla_con = chla_inversion(image, model=xgb_chla_model)
    TSS_con = TSS_inversion(image, model = xgb_tss_model)
    generate_geotiff_map(file_path, chla_con, 'chla_xgb.tif')
    # generate_chla_map(chla_con, output_path='cha_ml.png')
    # generate_tss_map(TSS_con, 'TSS_ml.png')
    # plt.show()

if __name__ == "__main__":
    main()