# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import os

import pandas as pd


# -------------------------- read -------------------------- #

def load_csv_columns(csv_path, name_ls, header=0, index_col=0):  # TODO: has not been tested
    df = pd.read_csv(csv_path, header=header, index_col=index_col)
    
    return [list(df[name]) for name in name_ls]


# -------------------------- write -------------------------- #

def save_to_csv(path, data, index=None, columns=None, dtype=None, copy=None, ):  # TODO: has not been tested
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataframe = pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy, )
    dataframe.to_csv(path, sep=',', )
    
    return True

def _():
    '''
    
    :return:
    '''

    # csv 格式
    # 读取文件
    import csv
    import numpy as np
    with open('./data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    data = np.array(rows)

    # 保存文件
    import pandas as pd
    a, b = [1, 2, 3], [4, 5, 6]  # 只能是一维数组
    dataframe = pd.DataFrame({
                                 'a_name': a,
                                 'b_name': b
                             })  # 字典中的key值即为csv中列名
    dataframe.to_csv("test.csv", index=False, sep=',')

    # xlsx 格式
    # 读取文件
    import pandas as pd
    df = pd.read_excel('./data/aa.xlsx', encoding='gbk')  # usecols =[0, 5] 指定列
    [num, item] = df.values.shape

    # 保存文件
    import pandas as pd
    data = pd.DataFrame([['a', 'b'], ['c', 'd']], index=['row 1', 'row 2'], columns=['col 1', 'col 2'])
    data.to_excel('./data/bb.xlsx', index=False)


if __name__ == '__main__':
    print('Hello World!')
    
    print('Brand-new World!')
