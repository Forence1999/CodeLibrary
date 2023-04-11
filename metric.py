# -*- coding:utf-8 _*-
# @license: MIT Licence
# @Author: Forence
# @Contact: wang00sheng@gmail.com
# @GitHub: https://github.com/Forence1999
# @Time: 4/10/2022
import numpy as np
from sklearn.metrics import confusion_matrix


def bca(y_true, y_pred):  # TODO: has not been tested
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def calculate_accuracy(y, y_pred, target_id=None):  # TODO: has not been tested
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        target_id: target class

    Returns:
        accuracy
        accuracy_nb: number of samples which are classified correctly
        target_rate:
        target_total: number of samples which changed their labels from others to target_id
    """
    y = checked_argmax(y, to_numpy=True)  # tf.argmax(y, axis=-1).numpy()
    y_pred = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    accuracy = np.mean(np.equal(y, y_pred))
    accuracy_nb = np.sum(np.equal(y, y_pred))
    if target_id is not None:
        non_target_idx = (y != target_id)
        target_total = np.sum((y_pred[non_target_idx] == target_id))
        target_rate = target_total / np.sum(non_target_idx)

        # Cases where non_target_idx is 0, so target_rate becomes nan
        if np.isnan(target_rate):
            target_rate = 1.  # 100% target num_adv for this batch

        return accuracy, accuracy_nb, target_rate, target_total
    else:
        return accuracy, accuracy_nb


def calculate_class_weighted_accuracy(y, y_pred, class_weight=None):  # TODO: has not been tested
    """
    Computes the accuracy as well as num_adv of attack of the target class.

    Args:
        y: ground truth labels. Accepts one hot encodings or labels.
        y_pred: predicted labels. Accepts probabilities or labels.
        class_weight: dictionary mapping class indices (integers) to a weight (float) value
    Returns:
        accuracy
        accuracy_nb: number of samples which are classified correctly
        target_rate:
        target_total: number of samples which changed their labels from others to target_id
    """
    y = checked_argmax(y, to_numpy=True)  # tf.argmax(y, axis=-1).numpy()
    y_pred = checked_argmax(y_pred, to_numpy=True)  # tf.argmax(y_pred, axis=-1).numpy()
    if class_weight is None:
        acc = np.mean(np.equal(y, y_pred))
        acc_num = np.sum(np.equal(y, y_pred))

        return {
            'acc'    : acc,
            'acc_num': acc_num
        }
    else:
        total = np.sum(list(class_weight.values()))
        for key in list(class_weight.keys()):
            class_weight[key] = class_weight[key] / total

        weighted_acc = 0
        for key in list(class_weight.keys()):
            key_idx = (y == key)
            key_total = np.sum((y_pred[key_idx] == key))
            weighted_acc += key_total / np.sum(key_idx) * class_weight[key]

        return weighted_acc


# -------------------------- utils -------------------------- #

def checked_argmax(y, to_numpy=False):  # TODO: has not been tested
    """
    Performs an argmax after checking if the input is either a tensor
    or a numpy matrix of rank 2 at least.

    Should be used in most cases for conformity throughout the
    codebase.

    Args:
        y: an numpy array or tensorflow tensor
        to_numpy: bool, flag to convert a tensor to a numpy array.

    Returns:
        an argmaxed array if possible, otherwise original array.
    """
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    if to_numpy:
        return np.array(y)
    else:
        return y


if __name__ == '__main__':
    print('Hello World!')

    print('Brand-new World!')
