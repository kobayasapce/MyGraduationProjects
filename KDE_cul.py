import pandas as pd
import numpy as np

def find_nearest_index(array, value):
    array = np.asarray(array)
    result = (np.abs(array - value)).argmin()
    return result

def cul_interval(value, confi):
    # 计算标称置信度
    std_confi = 100 - confi
    # 读取源数据，计算标准间隔
    data_path = "KDE result/datasets_raw_processed.csv"
    train_data = pd.read_csv(filepath_or_buffer=data_path).iloc[:,1]
    train_data = np.array(train_data).astype('float')
    x = np.linspace(start=train_data.min()-1, stop=train_data.max()+1, num=len(train_data)*5)
    # print(x.shape)
    spli = (train_data.max() - train_data.min())/(len(train_data)*5) # 这里的10与核密度估计部分参数需要保持一致
    # print(spli)

    # 读取面积数据
    x_prob_area_df = pd.read_csv("KDE result/KDE_prob_area.csv")["prob_area"]
    x_prob_area = np.asarray(x_prob_area_df).astype('float')
    # print(x_prob_area.shape)

    # 计算在数组x中与value最相近的数值下标
    index = find_nearest_index(array=x, value=value)
    area = x_prob_area[index]
    area_a = x_prob_area[index] - (std_confi/2) # 计算下界值
    area_b = x_prob_area[index] + (std_confi/2) # 计算上界值
    # 计算在数组x_prob_area中与area_a和area_b最相近的数值下标
    index_a = find_nearest_index(array=x_prob_area, value=area_a)
    index_b = find_nearest_index(array=x_prob_area, value=area_b)
    # 计算下界值
    a = value - (index - index_a)*spli
    # 计算上界值
    b = value + (index_b - index)*spli
    return a,b

