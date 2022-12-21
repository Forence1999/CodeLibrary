# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import os


# -------------------------- file -------------------------- #

def get_files_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            path = os.path.normpath(os.path.join(parent, f))
            if path.endswith(suffix):
                # img: (('.jpg', '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                file_list.append(path)
    return file_list


def get_files_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    file_list = []
    for parent, dirs, files in os.walk(root):
        for f in files:
            if f.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, f))
                file_list.append(path)
    return file_list


def get_subfiles_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    file_list = []
    for file_basename in os.listdir(root):
        fpath = os.path.join(root, file_basename)
        if os.path.isfile(fpath) and file_basename.endswith(suffix):
            # img: (('.jpg', '.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            path = os.path.normpath(fpath)
            file_list.append(path)
    return file_list


def get_subfiles_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    file_list = []
    for file_basename in os.listdir(root):
        fpath = os.path.join(root, file_basename)
        if os.path.isfile(fpath) and file_basename.startswith(prefix):
            path = os.path.normpath(fpath)
            file_list.append(path)
    return file_list


# -------------------------- dir -------------------------- #

def get_dirs_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            path = os.path.normpath(os.path.join(parent, d))
            if path.endswith(suffix):
                dir_list.append(path)
    return dir_list


def get_dirs_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    dir_list = []
    for parent, dirs, files in os.walk(root):
        for d in dirs:
            if d.startswith(prefix):
                path = os.path.normpath(os.path.join(parent, d))
                dir_list.append(path)
    return dir_list


def get_subdirs_by_suffix(root, suffix=''):
    if isinstance(suffix, str):
        suffix = (suffix,)
    else:
        suffix = tuple(suffix)
    
    dir_list = []
    for dir_basename in os.listdir(root):
        dpath = os.path.join(root, dir_basename)
        if os.path.isdir(dpath) and dir_basename.endswith(suffix):
            path = os.path.normpath(dpath)
            dir_list.append(path)
    return dir_list


def get_subdirs_by_prefix(root, prefix=''):
    if isinstance(prefix, str):
        prefix = (prefix,)
    else:
        prefix = tuple(prefix)
    
    dir_list = []
    for dir_basename in os.listdir(root):
        dpath = os.path.join(root, dir_basename)
        if os.path.isdir(dpath) and dir_basename.startswith(prefix):
            path = os.path.normpath(dpath)
            dir_list.append(path)
    return dir_list


# -------------------------- file & dir -------------------------- #

def add_prefix_and_suffix_4_basename(path, prefix=None, suffix=None):
    '''
    add prefix and/or suffix string(s) to a path's basename
    :param path:
    :param prefix:
    :param suffix:
    :return:
    '''
    dir_path, basename = os.path.split(path)
    filename, ext = os.path.splitext(basename)
    filename = str(prefix if prefix is not None else '') + filename + str(suffix if suffix is not None else '') + ext
    
    return os.path.join(dir_path, filename)


# -------------------------- read & save -------------------------- #
def _():
    '''
    
    :return:
    '''
    import numpy as np
    
    # txt 格式
    data = np.loadtxt("data.txt")
    np.save("data.txt", data)
    
    # 将文件读入list中
    data = []
    for line in open("data.txt", "r"):  # 设置文件对象并读取每一行文件
        data.append(line)  # 将每一行文件加入到list中
    
    # 将list写入txt文件
    with open('data.txt', 'w') as f:
        for name in info_list:
            f.write(name)
            f.write('\n')
    
    # mat 格式
    # 读取matlab的文件
    from scipy.io import loadmat
    
    def load_data(path_to_file):
        annots = loadmat(path_to_file)
        data = annots['x']
        labels = annots['y'].flatten()
        labels -= 1
        return data, labels
    
    # 保存matlab的文件
    from scipy.io import savemat
    
    savemat('D://data.mat', {
        'x': data
    })
    
    # npz 格式
    # 保存文件
    import numpy as np
    
    np.savez("data.npz", x=x, y=y)
    
    # 读取文件
    import numpy as np
    
    data = np.load('./data/data.npz')
    x, z = data['x'], data['z']
    
    # mkl 格式
    import torch
    
    # 读取模型文件
    model_path = './models/trained_model.pkl'
    model = MLP(n_feature=100, n_hidden=16, n_output=2)
    model.load_state_dict(torch.load(model_path))
    
    # 保存模型文件
    model = MLP(n_feature=100, n_hidden=16, n_output=2)
    torch.save(model.state_dict(), model_path)
    
    # pkl 格式
    import joblib
    
    # 模型训练
    gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
    
    # 模型存储
    joblib.dump(gbm, 'demo_model.pkl')
    
    # 模型加载
    gbm = joblib.load('demo_model.pkl')


def read_txt(path, encoding='utf-8'):
    '''
    read txt file
    :param path:
    :param encoding:
    :return:
    '''
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    return lines


if __name__ == '__main__':
    print('Hello World!')
    
    print('Brand-new World!')
