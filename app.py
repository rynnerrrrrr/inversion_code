from flask import Flask, request, render_template, send_file, jsonify
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import joblib


app = Flask(__name__)
UPLOAD_FOLDER = r'C:\Users\lijian\Desktop\CS_AI\CHL-A_TSS_inversion\static\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#加载已经保存好的模型
xgb_chla_model = joblib.load('xgb_chla_model.pkl')
xgb_tss_model = joblib.load('xgb_tss_model.pkl')

# 叶绿素a (Chl-a) 波段比值反演算法
def chla_inversion(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]
    
    green_div_blue = green / (blue + np.finfo(float).eps)  # 防止除零
    
    # 使用线性函数形式
    a = 6.25
    b = 4.126
    chla_concentration = a*green_div_blue - b
    return chla_concentration

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

def extract_and_process_bands(image, nodata_value=None):
    """
    提取图像的各个波段并进行标准化处理，同时处理无效值。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    nodata_value (float): 无效值的标识，默认为 None。
    
    返回:
    np.ndarray: 标准化后的像素特征数组。
    """
    blue = image[:, :, 0] / 10000
    green = image[:, :, 1] / 10000
    red = image[:, :, 2] / 10000
    nir = image[:, :, 3] / 10000
    
    # 处理无效值
    if nodata_value is not None:
        valid_mask = ~np.any(image == nodata_value, axis=-1)
        blue[~valid_mask] = np.nan
        green[~valid_mask] = np.nan
        red[~valid_mask] = np.nan
        nir[~valid_mask] = np.nan
    else:
        valid_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    
    # 将图像数据重塑为二维数组，每一行代表一个像素的特征向量
    height, width = image.shape[:2]
    pixel_features = np.stack([blue, green, red, nir], axis=-1).reshape(-1, 4)
    
    # 过滤掉包含无效值的像素
    if nodata_value is not None:
        valid_features = pixel_features[~np.any(np.isnan(pixel_features), axis=1)]
    else:
        valid_features = pixel_features
    
    standardized_features = standardize_data(valid_features)
    
    return standardized_features, height, width, valid_mask

# def extract_and_process_bands(image):
#     """
#     提取图像的各个波段并进行标准化处理。
    
#     参数:
#     image (np.ndarray): 输入的图像数据。
    
#     返回:
#     np.ndarray: 标准化后的像素特征数组。
#     """
#     blue = image[:, :, 0] / 10000    #除以10000是因为影像数据没有进行相应的大气校正
#     green = image[:, :, 1] / 10000
#     red = image[:, :, 2] / 10000
#     nir = image[:, :, 3] / 10000
    
#     # 将图像数据重塑为二维数组，每一行代表一个像素的特征向量
#     height, width = image.shape[:2]
#     pixel_features = np.stack([blue, green, red, nir], axis=-1).reshape(-1, 4)
#     standardized_features = standardize_data(pixel_features)
    
#     return standardized_features, height, width

#影像中不存在无效值的chla反演方法
def chla_inversion_ml(image, model):
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

#影像中存在无效值(-9999)的chla反演方法
def chla_inversion(image, model, nodata_value=-9999):
    """
    使用机器学习模型进行叶绿素a浓度的反演，同时处理无效值。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    model: 预训练的机器学习模型。
    nodata_value (float): 无效值的标识，默认为 None。
    
    返回:
    np.ndarray: 预测的叶绿素a浓度图像。
    """
    # 提取并处理图像波段
    pixel_features, height, width, valid_mask = extract_and_process_bands(image, nodata_value=-9999)
    
    # 使用模型进行批量预测
    chla_concentration = model.predict(pixel_features)
    
    # 将预测结果重塑为图像的形状
    chla_image = np.full((height, width), nodata_value)
    valid_indices = np.where(valid_mask.ravel())[0]
    chla_image.ravel()[valid_indices] = chla_concentration
    
    return chla_image

#影像中不存在无效值的TSS机器学习反演
def TSS_inversion_ml(image, model):
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

#影像中存在无效值的TSS机器学习反演
def TSS_inversion(image, model, nodata_value=-9999):
    """
    使用机器学习模型进行悬浮物浓度的反演。
    
    参数:
    image (np.ndarray): 输入的图像数据。
    model: 预训练的机器学习模型。
    
    返回:
    np.ndarray: 预测的TSS浓度图像。
    """
    # 提取并处理图像波段
    pixel_features, height, width, valid_mask = extract_and_process_bands(image, nodata_value)
    
    # 使用模型进行批量预测
    TSS_concentration = model.predict(pixel_features)
    
    # 将预测结果重塑为图像的形状
    TSS_image = np.full((height, width), nodata_value)
    valid_indices = np.where(valid_mask.ravel())[0]
    TSS_image.ravel()[valid_indices] = TSS_concentration
    
    return TSS_image

# TSS波段比值反演算法
def tss_inversion(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    nir = image[:, :, 3]
    

    # 使用某种算法计算TSS浓度
    tss_concentration = 2.9129*np.exp(38.026*red/1000000)  # 示例算法
    return tss_concentration


# 读取影像数据
def read_image(file_path):
    dataset = gdal.Open(file_path)
    image = dataset.ReadAsArray()
    for band_num in range(1, 5):
        # 读取当前波段的像素值
        band = dataset.GetRasterBand(band_num)
        band_values = band.ReadAsArray().reshape(-1)

        # 计算0和-9999的数量
        zero_count = np.sum(band_values == 0)
        minus_9999_count = np.sum(band_values == -9999)
        print(f"Band {band_num}: 0 count = {zero_count}, -9999 count = {minus_9999_count}")
    
    image = np.transpose(image, (1, 2, 0))
    return image

# 生成Chl-a浓度分布图
def generate_chla_map(chla_concentration, output_path, nodata_value=-9999):
    """
    生成叶绿素a浓度分布图，并忽略无效值。
    
    参数:
    chla_concentration (np.ndarray): 叶绿素a浓度数组。
    output_path (str): 输出图像的路径。
    nodata_value (float): 无效值的标识，默认为 -9999。
    """
    # 创建一个掩码数组，用于屏蔽无效值
    mask = chla_concentration == nodata_value
    
    # 创建一个 masked 数组
    masked_chla = np.ma.masked_where(mask, chla_concentration)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(masked_chla, cmap='viridis', vmin=np.min(masked_chla), vmax=np.max(masked_chla))
    plt.colorbar(label='Chl-a Concentration')
    plt.title('Chl-a Concentration Map')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()  

#生成TSS浓度分布图
def generate_tss_map(tss_concentration, output_path, nodata_value=-9999):
    """
    生成TSS浓度分布图，并忽略无效值。
    
    参数:
    TSS_concentration (np.ndarray): Tss浓度数组。
    output_path (str): 输出图像的路径。
    nodata_value (float): 无效值的标识，默认为 -9999。
    """
    # 创建一个掩码数组，用于屏蔽无效值
    mask = tss_concentration == nodata_value
    
    # 创建一个 masked 数组
    masked_tss = np.ma.masked_where(mask, tss_concentration)


    plt.figure(figsize=(10, 10))
    plt.imshow(masked_tss, cmap='viridis', vmin=np.min(masked_tss), vmax=np.max(masked_tss))
    plt.colorbar(label='TSS Concentration')
    plt.title('TSS Concentration Map')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        image = read_image(filename)
        chla_concentration = chla_inversion(image, model=xgb_chla_model, nodata_value=-9999)
        tss_concentration = TSS_inversion(image, model=xgb_tss_model, nodata_value=-9999)

        chla_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chla_map.png')
        tss_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'tss_map.png')
        
        generate_chla_map(chla_concentration, chla_output_path, nodata_value=-9999)
        generate_tss_map(tss_concentration, tss_output_path, nodata_value=-9999)
        
        return jsonify({
            "chla_map_url": f"/download?file=chla_map.png",
            "tss_map_url": f"/download?file=tss_map.png"
        })
        
        
    
@app.route('/download')
# def download_file():
#     path = os.path.join(app.config['UPLOAD_FOLDER'], 'chla_map.png')
#     return send_file(path, as_attachment=True)
def download_file():
    file_name = request.args.get('file')
    path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
