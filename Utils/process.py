import torch
import numpy as np

"""
Grid locator Function
"""


def locate_grid_mult(x, y):
    """

    :param x: x coordinate to be mapped
    :param y: y coordinate to be mapped
    :return: grid_x: x coordinate after mapping it to its equivalent grid
    :return: grid_y: y coordinate after mapping it to its equivalent grid
    """
    if torch.sum(torch.less(x, 0)):
        x[x < 0] = 0
    if torch.sum(torch.less(y, 0)):
        y[y < 0] = 0
    if torch.sum(torch.greater(x, 512)):
        x[x > 512] = 512
    if torch.sum(torch.greater(y, 512)):
        y[y > 512] = 512
    grid_x = torch.divide(x, 32).type(torch.int)
    grid_y = torch.divide(y, 32).type(torch.int)
    grid_x[grid_x == 16] = 15
    grid_y[grid_y == 16] = 15

    return grid_x, grid_y


"""## Convert Marking Point to Tensor"""


def marking_to_tensor(marking_points):
    """
    :param marking_points: marking points to be converted to tensor

    :return: output_tensor: tensor of marking points
    """
    output_tensor = torch.zeros((5, marking_points[0].shape[0]))
    for i in range(5):
        output_tensor[i] = marking_points[i]
    return output_tensor


"""## Remove Row from Tensor"""


def remove_row_ls(tens, row_index_ls):
    """
    :param tens: tensor to remove rows from
    :param row_index_ls: index list of rows to be removed
    :return: tensor after removing rows
    """
    tens_copy = torch.clone(tens)
    ls = []
    for i in range(tens_copy.shape[0]):
        ls.append(i)
    ls = torch.tensor(ls)
    for i in range(len(row_index_ls)):
        index = ls[ls != row_index_ls[i]]
        ls = index
    index = torch.tensor(index)
    return torch.index_select(tens_copy, 0, index)


"""#Calculate angle between two points"""


def calc_angle_2(p1, p2):
    """
    :param p1: first point

    :param p2: second point
    :return: theta: angle between first point and second point
    """
    y_diff = p2[1] - p1[1]
    x_diff = p2[0] - p1[0]
    theta = np.zeros_like(x_diff)
    theta = np.arctan2(y_diff, x_diff)
    return theta


"""# Complete marking vector (6x16x16)"""


def complete_marking_vector_label_mult(training_examples):
    """
    :param training_examples: training examples to create target label for them

    :return: label_vector: target label of passed training examples
    """
    number_of_trainig_examples = training_examples['image'].shape[0]
    label_vector = torch.zeros((number_of_trainig_examples, 6, 16, 16), dtype = torch.float64)

    label_vector[:, 0, :, :] = label_vector[:, 0, :, :] - 0.5

    for i in range(len(training_examples['marks'])):
        grid_x, grid_y = locate_grid_mult((training_examples['marks'][i][0]),
                                          (training_examples['marks'][i][1]))
        for j in range(number_of_trainig_examples):
            label_vector[j, 1:, grid_x[j], grid_y[j]] = (marking_to_tensor(training_examples['marks'][i]))[:, j]
            label_vector[j, 0, grid_x[j], grid_y[j]] = 0.5  # confidence

    label_vector[:, 5, :, :] = (label_vector[:, 5, :, :]) - 0.5  # shape

    x_old = torch.clone(label_vector[:, 1, :, :])
    y_old = torch.clone(label_vector[:, 2, :, :])
    x_val = (torch.clone(label_vector[:, 1, :, :]) % 32) - 16
    y_val = (torch.clone(label_vector[:, 2, :, :]) % 32) - 16
    dir_x = torch.clone(label_vector[:, 3, :, :]) - (x_old - x_val)
    dir_y = torch.clone(label_vector[:, 4, :, :]) - (y_old - y_val)

    marking_direction = (calc_angle_2([x_val, y_val], [dir_x, dir_y]))
    cos_theta = torch.cos(marking_direction)   # cos
    sin_theta = torch.sin(marking_direction)   # sin

    label_vector[:, 1, :, :] = ((label_vector[:, 1, :, :] % 32) - 16) / 32  # x
    label_vector[:, 2, :, :] = ((label_vector[:, 2, :, :] % 32) - 16) / 32  # y
    label_vector[:, 3, :, :] = torch.clone(cos_theta) * 0.5
    label_vector[:, 4, :, :] = torch.clone(sin_theta) * 0.5

    return label_vector
