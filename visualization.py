# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from PIL import Image
from matplotlib.collections import QuadMesh
from pandas import DataFrame
from sklearn.metrics import confusion_matrix


# -------------------------- single -------------------------- #

def plot_curve(data, title=None, img_path=None, show=True, y_lim=None, linestyle='-', linewidth=1):
    '''
    data: tuple of every curve's label, data and color
    for example:
        name = ['Training Acc', 'Validation Acc', 'Test Acc']
        data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        plot_curve(data=list(zip(name, data, color)), title=title, img_path=img_path)
    '''
    plt.figure()
    for i in data:
        x_len = len(i[1])
        x = list(range(0, x_len))
        plt.plot(x, i[1], i[2], label=i[0], linestyle=linestyle, linewidth=linewidth)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_hist(data, title=None, img_path=None, bins=100, show=True):
    '''
    data: tuple of every curve's label, data and color
    for example:
        name = ['Training Acc', 'Validation Acc', 'Test Acc']
        data = [train_acc, val_acc, test_acc]
        color = ['r', 'y', 'cyan']
        plot_curve(data=list(zip(name, data, color)), title=title, img_path=img_path)

    '''
    plt.figure()
    for i in data:
        plt.hist(i[1], bins, color=i[2], label=i[0])
    
    # plt.ylim(0, 1.1)
    plt.title(title)
    plt.legend()
    if img_path is not None:
        plt.savefig(img_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_bars(data, color=None, title=None, x_labels=None, y_label=None, y_lim=None, tick_step=1., group_gap=0.2,
                    bar_gap=0., plt_show=True, value_show=True, dpi=300, value_fontsize=5, value_interval=0.01,
                    value_format='%.2f', save_path=None):
    '''
    x_labels: x轴坐标标签序列
    data: 二维列表，每一行为同一颜色的各个bar值，每一列为同一个横坐标的各个bar值
    tick_step: 默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap: 组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组中柱子间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''
    data = np.asarray(data)
    bar_per_group_num = len(data[0])
    if color is not None:
        assert len(color) >= bar_per_group_num
    x_ticks = np.arange(bar_per_group_num) * tick_step
    group_num = len(data)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num  # 组内每个bar的宽度
    bar_width = bar_span - bar_gap
    baseline_x = x_ticks - (group_width - bar_span) / 2  # baseline_x为每组柱子第一个柱子的基准x轴位置
    plt.figure(dpi=dpi)
    for index, y in enumerate(data):
        x = baseline_x + index * bar_span
        if color is not None:
            plt.bar(x, y, bar_width, color=color[index])
        else:
            plt.bar(x, y, bar_width)
        if value_show:
            for x0, y0 in zip(x, y):
                plt.text(x0, y0 + value_interval, value_format % y0, ha='center', va='bottom', fontsize=value_fontsize)
    if title is not None:
        plt.title(title)
    if x_labels is not None:
        plt.xticks(x_ticks, x_labels)
    if y_label is not None:
        plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if save_path is not None:
        plt.savefig(save_path)
    if plt_show:
        plt.show()


def plot_2d_heatmap(data, is_confusion_matrix=False, title=None, x_label=None, x_ticks=None, y_label=None, y_ticks=None,
                    cmap='coolwarm', data_format='%.2f', fontsize=10, linewidth=0.5, show_cbar=True, figsize=None,
                    show_null_value=True, annot_cell=True, insert_summary=True, use_mean=True, show_percent=False,
                    tick_rotation_degree=None):
    '''
    print 2D matrix
    Args:
        data: 2d list or array
        columns:
        cmap: Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fmt:
        cbar:
        show_null_value:
        annot_cell (bool): annotate values in the cells
        show_percent: False: only show digit; True: only show percent (%)
        insert_summary: show summary cow and column
        use_mean: for summarization (mean or sum)
    Returns:

    '''
    
    def config_cell_4_confusion_matrix(data, position, oText, facecolors, fontsize, data_format='%.2f',
                                       show_null_value=0, ):
        # TODO have not been implemented yet
        """
            config cell text and colors
                and return text elements to add and to del
        """
        text_add = []
        text_del = []
        column, row = list(map(int, position))
        cell_val = data[row][column]
        tot_all = data[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = data[:, column]
        ccl = len(curr_column)
        
        # last row  and/or last column
        if (column == (ccl - 1)) or (row == (ccl - 1)):
            # tots and percents
            if (cell_val != 0):
                if (column == ccl - 1) and (row == ccl - 1):
                    tot_rig = 0
                    for i in range(data.shape[0] - 1):
                        tot_rig += data[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (column == ccl - 1):
                    tot_rig = data[row][row]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif (row == ccl - 1):
                    tot_rig = data[column][column]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0
            
            per_ok_s = ['%.2f%%' % (per_ok), '100%'][int(per_ok == 100)]
            
            # text to DEL
            text_del.append(oText)
            
            # text to ADD
            font_prop = fm.FontProperties(weight='bold', size=fontsize)
            text_kwargs = dict(color='black', ha="center", va="center", gid='sum', fontproperties=font_prop)
            lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy()
            dic['color'] = 'g'
            lis_kwa.append(dic)
            dic = text_kwargs.copy()
            dic['color'] = 'r'
            lis_kwa.append(dic)
            lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
                # print 'row: %s, column: %s, newText: %s' %(row, column, newText)
                text_add.append(newText)
            # print '\n'
            
            # set background color for sum cells (last row and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if (column == ccl - 1) and (row == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            facecolors[posi] = carr
        
        else:
            if (per > 0):
                txt = '%s\n%.2f%%' % (cell_val, per)
            else:
                if (show_null_value == 0):
                    txt = ''
                elif (show_null_value == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            oText.set_text(txt)
            
            # main diagonal
            if (column == row):
                # set color of the textin the diagonal to white
                oText.set_color('black')
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color('r')
        
        return text_add, text_del
    
    def config_cell(data, position, oText, fontsize, data_format, show_null_value=False, show_percent=False):
        """
            config cell text and colors
                and return text elements to add and to del
        """
        text_add = []
        text_del = []
        column, row = list(map(int, position))
        cell_val = data[row][column]
        
        text_del.append(oText)
        
        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fontsize)
        text_kwargs = dict(color='black', ha="center", va="center", gid='sum', fontproperties=font_prop)
        
        if show_percent:
            text_ls = ['', ] if ((show_null_value == False) and np.allclose(cell_val, 0)) \
                else [str(data_format + '%%') % (cell_val * 100.)]
            kwargs_ls = [text_kwargs]
            posi_ls = [(oText._x, oText._y)]
        else:
            text_ls = ['', ] if ((show_null_value == False) and (cell_val == 0)) else [data_format % cell_val, ]
            kwargs_ls = [text_kwargs] * len(text_ls)
            posi_ls = [(oText._x, oText._y)]
        
        for i in range(len(text_ls)):
            newText = dict(x=posi_ls[i][0], y=posi_ls[i][1], text=text_ls[i], kw=kwargs_ls[i])
            text_add.append(newText)
        
        return text_add, text_del
    
    def insert_totals(data_df, use_mean=False):
        """ insert summary row and column """
        
        if use_mean:
            data_df['Mean'] = data_df.mean(axis=1)
            avg_line = data_df.mean(axis=0)
            avg_line.name = 'Mean'
            data_df = data_df.append(avg_line, ignore_index=False)
        else:
            data_df['Sum'] = data_df.sum(axis=1)
            sum_line = data_df.sum(axis=0)
            sum_line.name = 'Sum'
            data_df = data_df.append(sum_line, ignore_index=False)
        
        return data_df
    
    data = np.array(data)
    x_ticks = map(str, range(data.shape[1])) if (x_ticks is None) else x_ticks
    y_ticks = map(str, range(data.shape[0])) if (y_ticks is None) else y_ticks
    data_df = DataFrame(data, index=y_ticks, columns=x_ticks)
    
    if insert_summary:
        data_df = insert_totals(data_df, use_mean=use_mean)
        data = np.asarray(data_df)
    
    # create plot
    fig = plt.figure(figsize)
    ax = fig.gca()  # Get Current Axis
    ax.cla()  # clear existing plot
    
    # annot_kws = {
    #     "size": fontsize
    # }
    ax = sn.heatmap(data_df, annot=annot_cell, linewidths=linewidth, ax=ax, cbar=show_cbar,  # annot_kws=annot_kws,
                    cmap=cmap, linecolor='w', )
    if tick_rotation_degree is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=fontsize)
    
    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()
    
    # iter in text elements
    text_add = []
    text_del = []
    for idx, text in enumerate(ax.collections[0].axes.texts):
        posi = np.array(text.get_position()) - [0.5, 0.5]
        
        if is_confusion_matrix:
            txt_res = config_cell_4_confusion_matrix(data=data, position=posi, oText=text, fontsize=fontsize,
                                                     data_format=data_format, show_null_value=show_null_value,
                                                     facecolors=facecolors, )
        else:
            txt_res = config_cell(data=data, position=posi, oText=text, fontsize=fontsize, data_format=data_format,
                                  show_null_value=show_null_value, show_percent=show_percent, )
        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])
    
    for item in text_del:
        item.remove()
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])
    
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.tight_layout()  # set layout slim
    plt.show()


def plot_confusion_matrix_from_y(y, y_pred, x_label=None, x_ticks=None, y_label=None, y_ticks=None,
                                 cmap="Spectral", data_format='%.2f', fontsize=10, linewidth=0.5, show_cbar=True,
                                 figsize=None, show_null_value=False, annot_cell=True, insert_summary=True,
                                 use_mean=True):
    # TODO not implemented completely yet
    """
        plot confusion matrix function with y and y_pred whitout a confusion matrix
    """
    
    c_matrix = confusion_matrix(y, y_pred)
    
    plot_2d_heatmap(data=c_matrix, is_confusion_matrix=False, x_label=x_label, x_ticks=x_ticks, y_label=y_label,
                    y_ticks=y_ticks, show_percent=True,
                    cmap=cmap, data_format=data_format, fontsize=fontsize, linewidth=linewidth, show_cbar=show_cbar,
                    figsize=figsize, show_null_value=show_null_value, annot_cell=annot_cell,
                    insert_summary=insert_summary, use_mean=use_mean)


def plot_raw(clean, adv, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(clean), np.min(clean)
        clean = (clean - min_) / (max_ - min_)
        adv = (adv - min_) / (max_ - min_)
    
    plt.figure()
    x = np.arange(clean.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, adv[0] - np.mean(adv[0]), linewidth=2.0, color='red', label='Adversarial sample')  # plot adv data
    l2, = plt.plot(x, clean[0] - np.mean(adv[0]), linewidth=2.0, color='dodgerblue',
                   label='Original sample')  # plot clean data
    for i in range(1, 5):
        plt.plot(x, adv[i] + i - np.mean(adv[i]), linewidth=2.0, color='red')  # plot adv data
        plt.plot(x, clean[i] + i - np.mean(adv[i]), linewidth=2.0, color='dodgerblue')  # plot clean data
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l2, l1], labels=['Original sample', 'Poisoned sample'], loc='upper right', ncol=2,
               fontsize=10)
    # plt.savefig(file_name + '.eps')
    plt.show()


def plot_signal(signal, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(signal), np.min(signal)
        signal = (signal - min_) / (max_ - min_)
    
    plt.figure()
    x = np.arange(signal.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, signal[0] - np.mean(signal[0]), linewidth=2.0, color='red',
                   label='Adversarial sample')  # plot adv data
    
    for i in range(1, 5):
        plt.plot(x, signal[i] + i - np.mean(signal[i]), linewidth=2.0, color='red')  # plot adv data
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 5.0])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y, y_names, fontsize=10)
    plt.legend(handles=[l1], labels=['Original sample'], loc='upper right', ncol=2,
               fontsize=10)
    plt.savefig(file_name + '.eps')


def plot_pulse(pulse, file_name):
    plt.figure()
    channels = pulse.shape[0]
    x = np.arange(pulse.shape[1]) * 1.0 / 256
    l1, = plt.plot(x, pulse[0], linewidth=2.0, color='g', label='pulse')  # plot adv data
    for i in range(1, 5):
        plt.plot(x, pulse[i] + i * 1.2, linewidth=2.0, color='g')  # plot adv data
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylim([-0.5, 6.5])
    temp_y = np.arange(5)
    y_names = ['Channel {}'.format(int(y_id)) for y_id in temp_y]
    plt.yticks(temp_y * 1.2, y_names, fontsize=10)
    # plt.legend(handles=[l1], labels=['Adversarial sample'], loc='upper right', ncol=2,
    #            fontsize=10)
    plt.savefig(file_name + '.eps')


def show_as_image(mask, name='mask.eps'):
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = 1 - mask
    # x = np.arange(mask.shape[1]) * 1.0 / 256
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig(name)


# -------------------------- multiple -------------------------- #

def img_splice(img_paths, save_path, sgl_img_size):
    '''
    img_paths: 2-D list storing the paths of images
    sgl_img_size: size of single image
    '''
    
    width, height = sgl_img_size
    nb_column = max([len(i) for i in img_paths])
    nb_row = len(img_paths)
    res_img = Image.new(mode='RGB', size=(width * nb_column, height * nb_row), color=(255, 255, 255))
    for i in range(len(img_paths)):
        for j in range(len(img_paths[i])):
            # load imgs
            img = Image.open(img_paths[i][j])
            
            res_img.paste(img, (width * j, height * (i),
                                width * (j + 1), height * (i + 1)))
    res_img.save(save_path)
    return res_img


if __name__ == '__main__':
    """ test function with y_true and y_test """
    y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ])
    y_test = np.array([1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, ])
    
    plot_confusion_matrix_from_y(y_true, y_test, use_mean=True)
