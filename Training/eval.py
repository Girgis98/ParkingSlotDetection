from Training.training import device, data_loader
from Utils.process import complete_marking_vector_label_mult
import torch

from model import MarkingPointDetector


def eval(output, label):
    '''
    Evaluate Model after training
    :param output: Model Prediction
    :type output: 6x16x16 tensor
    :param label: Objective
    :type label: 6x16x16 tensor
    :return: accuracy, f1_score, precision, recall
    :rtype: float64
    '''
    conf_flag = (label[:, 0, :, :] == 0.5)
    not_conf_flag = (label[:, 0, :, :] == -0.5)
    out = output.permute(0, 2, 3, 1)
    label_p = label.permute(0, 2, 3, 1)
    threhold_conf = 0.2
    threhold_coordinates = 0.3125
    therhold_direction_sin = 0.4
    therhold_direction_cos = 0.92

    true_negative = torch.sum(torch.bitwise_not(
        torch.bitwise_xor((out[not_conf_flag][:, 0] < threhold_conf), label_p[not_conf_flag][:, 0] == -0.5)).type(
        torch.float)).item()

    false_negative = torch.sum(
        (torch.bitwise_xor((out[not_conf_flag][:, 0] < threhold_conf), label_p[not_conf_flag][:, 0] == -0.5)).type(
            torch.float)).item()

    confidence_ones = torch.bitwise_not(
        torch.bitwise_xor((out[conf_flag][:, 0] >= threhold_conf), label_p[conf_flag][:, 0] == 0.5))
    x_val = (torch.abs(out[conf_flag][:, 1] - label_p[conf_flag][:, 1]) <= threhold_coordinates)
    y_val = (torch.abs(out[conf_flag][:, 2] - label_p[conf_flag][:, 2]) <= threhold_coordinates)
    x_dir_val = (torch.abs(out[conf_flag][:, 3] - label_p[conf_flag][:, 3]) <= therhold_direction_cos)
    y_dir_val = (torch.abs(out[conf_flag][:, 4] - label_p[conf_flag][:, 4]) <= therhold_direction_sin)
    shape_val = torch.bitwise_not(torch.bitwise_xor((out[conf_flag][:, 5] >= -0.05), label_p[conf_flag][:, 5] == 0.5))

    total_positive = (label_p[conf_flag][:, 0]).shape[0]

    '''Calculation Details'''
    # c = torch.sum(confidence_ones).item()
    # x = torch.sum(x_val).item()
    # y = torch.sum(y_val).item()
    # dx = torch.sum(x_dir_val).item()
    # dy = torch.sum(y_dir_val).item()
    # s = torch.sum(shape_val).item()
    #
    # print("total positive:", total_positive)
    # print("conf:", c)
    # print("x:", x)
    # print("y:", y)
    # print("dx:", dx)
    # print("dy:", dy)
    # print("shape:", s)

    true_positive = torch.sum(torch.bitwise_and(confidence_ones, torch.bitwise_and
    (x_val, torch.bitwise_and(y_val, torch.bitwise_and(x_dir_val, torch.bitwise_and(y_dir_val, shape_val))))).type(
        torch.float)).item()

    false_positive = (total_positive - true_positive)

    accuracy = (true_negative + true_positive) / (true_negative + false_negative + true_positive + false_positive) if (
                                                                                                                              true_negative + false_negative + true_positive + false_positive) != 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (recall * precision) / (recall + precision) if (recall + precision) != 0 else 0
    print("TP:", true_positive, " TN:", true_negative, " FP:", false_positive, " FN:", false_negative)
    return accuracy, f1_score, precision, recall
