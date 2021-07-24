import torch
import math

"""
## Calculate Direction(angle)
"""


def calc_angle(p1, p2):
    y_diff = p2[1] - p1[1]
    x_diff = p2[0] - p1[0]
    if x_diff == 0 and y_diff == 0:
        return 0
    theta = math.atan(abs(y_diff / x_diff)) * (180 / math.pi)

    if x_diff >= 0 and y_diff >= 0:
        return theta
    elif x_diff < 0 and y_diff >= 0:
        return 180 - theta
    elif x_diff < 0 and y_diff < 0:
        return 180 - theta
    else:
        return theta


"""## Calculate difference in direction"""


def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2 * math.pi - diff


"""##Calculate absolute difference in direction"""


def abs_direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(abs(direction_a) - abs(direction_b))
    return diff


"""## check a point is between two certain points"""


def isBetween(a, b, c):
    epsilon = 0.001
    crossproduct = (c[2] - a[2]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[2] - a[2])
    print("crossproduct", crossproduct)
    # compare versus epsilon for floating point values, or != 0 if using integers
    # sin 0 =0 , check theta >0 (not parallel)
    if abs(crossproduct) > epsilon:
        return False

    # cos 180 = -1, check that they are in the same direction
    dotproduct = (c[1] - a[1]) * (b[1] - a[1]) + (c[2] - a[2]) * (b[2] - a[2])
    print("dotproduct", dotproduct)
    if dotproduct < 0:
        return False

    squaredlengthba = (b[1] - a[1]) * (b[1] - a[1]) + (b[2] - a[2]) * (b[2] - a[2])
    print("squaredlengthba", squaredlengthba)

    if dotproduct > squaredlengthba:
        return False

    return True


"""## Remove Row from Tensor"""


def remove_row_ls(tens, row_index_ls):
    tens_copy = torch.clone(tens)
    ls = []
    for i in range(tens_copy.shape[0]):
        ls.append(i)
    ls = torch.tensor(ls)
    # index = []
    for i in range(len(row_index_ls)):
        index = ls[ls != row_index_ls[i]]
        ls = index
    index = torch.tensor(index)
    return torch.index_select(tens_copy, 0, index)


"""## Calculate Distance between two points"""


def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    x_a = point_a[1]
    y_a = point_a[2]
    x_b = point_b[1]
    y_b = point_b[2]

    distx = x_a - x_b
    disty = y_a - y_b
    dist = math.sqrt(math.pow(distx, 2) + math.pow(disty, 2))
    return dist


"""### Order points pair clockwise"""


def order_pair(point1, point2):
    if abs(point2[2] - point1[2]) < 20:
        if point2[1] > point1[1]:
            return point2, point1
        else:
            return point1, point2
    else:
        if max(point2[1], point1[1]) > 255:
            if point2[2] < point1[2]:
                return point2, point1
            else:
                return point1, point2
        elif max(point2[1], point1[1]) < 255:
            print('first half')
            if point1[2] > point2[2]:
                print('1,2')
                return point1, point2
            else:
                return point2, point1


"""## calculate angle vector form a point"""


def calc_angle_vector(p1):
    y_diff = p1[1]
    x_diff = p1[0]
    if x_diff == 0 and y_diff == 0:
        return 0
    theta = math.atan(abs(y_diff / x_diff)) * (180 / math.pi)
    if x_diff >= 0 and y_diff >= 0:
        return theta
    elif x_diff < 0 and y_diff >= 0:
        return 180 - theta
    elif x_diff < 0 and y_diff < 0:
        return 180 + theta
    elif x_diff > 0 and y_diff < 0:
        return -1 * theta
    else:
        return theta
