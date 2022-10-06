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


if __name__ == '__main__':
    print('Hello World!')
    
    print('Brand-new World!')
