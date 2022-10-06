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


if __name__ == '__main__':
    print('Hello World!')
    
    print('Brand-new World!')
