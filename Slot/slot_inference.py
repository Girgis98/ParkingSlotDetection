"""
Mapper
* Maps the predicted output into : confidence , x (from 0 to 512) , y (from 0 to 512) , direction-x (from 0 to 512) , direction-y (from 0 to 512) , shape of marking point
"""
import torch
import numpy as np
import math

#from Training.training import device
from Utils.new_utensils_slotinference import calc_angle_vector, calc_angle, abs_direction_diff, direction_diff, \
    calc_point_squre_dist
from Utils.process import remove_row_ls

device = torch.device("cpu")
def mapper_vectorized(prediction_in):
    prediction = torch.clone(prediction_in)
    prediction[:, 1:3, :, :] = (prediction[:, 1:3, :, :] * 32) + 16
    values = torch.arange(0, 512, 32)
    xv, yv = torch.meshgrid([values, values])
    yv = yv.to(device)
    xv = xv.to(device)

    prediction[:, 1, :, :] = prediction[:, 1, :, :] + xv
    prediction[:, 2, :, :] = prediction[:, 2, :, :] + yv
    cos_value = prediction[:, 3, :, :] * 2
    sin_value = prediction[:, 4, :, :] * 2
    x_val_mapped = prediction[:, 1, :, :]
    y_val_mapped = prediction[:, 2, :, :]
    x_dir = x_val_mapped + (40 * cos_value)
    y_dir = y_val_mapped + (40 * sin_value)

    prediction[:, 3, :, :] = x_dir
    prediction[:, 4, :, :] = y_dir
    prediction[:, 0, :, :] = (prediction[:, 0, :, :] + 0.5) * 100
    prediction[:, 5, :, :] = (prediction[:, 5, :, :] >= 0)

    return prediction


"""# Prediction"""


def predict(image, model, device):
    out = model(image).to(device)
    out = out.reshape((-1, 6, 16, 16))
    out = mapper_vectorized(out)
    return out


"""## Remove points lower than Thresthold 
takes the predicted point and the threshold, check if the confidence value is less than this threshold then we remove this point
"""


def get_predicted_points(prediction, thresh):  # prediction (training,6,16,16)
    """Get marking points from one predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    if len(prediction.shape) == 3:
        prediction = prediction.reshape((1, 6, 16, 16))
    num_training_examples = prediction.shape[0]
    predicted_points = []
    prediction = prediction.detach().cpu()

    index = (torch.greater_equal(prediction[:, 0, :, :], thresh))

    prediction = prediction.permute(0, 2, 3, 1)

    for i in range(num_training_examples):
        predicted_points.append(prediction[i, index[i, :, :]])

    predicted_points_copy = []
    for i in range(len(predicted_points)):
        predicted_points_copy.append(torch.clone(predicted_points[i]))

    for i in range(len(predicted_points)):  # on training examples
        for j in range(len(predicted_points[i])):  # on points
            if predicted_points[i][j][1] < 10 and predicted_points[i][j][
                2] < 10:  # remove points with negative x , y values
                predicted_points_copy[i] = torch.cat(
                    [predicted_points_copy[i][0:j, :], predicted_points_copy[i][j + 1:, :]])
            if predicted_points[i][j][1] > 511 or predicted_points[i][j][
                2] > 511:  # remove points with negative x , y values
                print('here', predicted_points[i])
                predicted_points_copy[i] = torch.cat(
                    [predicted_points_copy[i][0:j - 1, :], predicted_points_copy[i][j:, :]])
    return predicted_points_copy


"""## Non-max suppression for near points
take predicted points and check if there is more than one point close to each other it will remove all of the except the one which have the highest confidence
"""


def non_maximum_suppression(pred_points):  # pred_points (training,num_points,6) (list)
    """Perform non-maxmum suppression on marking points."""
    pred_copy = pred_points
    threshold_near = 40
    for k, tens in enumerate(pred_copy):
        index_arr = []
        for i in range(tens.shape[0]):
            for j in range(i + 1, tens.shape[0]):
                i_x = tens[i][1]
                i_y = tens[i][2]
                j_x = tens[j][1]
                j_y = tens[j][2]
                abs_x = abs(j_x - i_x)
                abs_y = abs(j_y - i_y)
                abs_dis = math.sqrt(math.pow(abs_x, 2) + math.pow(abs_y, 2))
                if abs_dis <= threshold_near:
                    idx = i if tens[i][0] < tens[j][0] else j
                    index_arr.append(idx)
        if len(index_arr) != 0:
            pred_copy[k] = remove_row_ls(tens, index_arr)
    return pred_copy


"""## Check the third point
function checks if the line passes between the two marking points contains another third point between them or not
"""


def pass_through_third_point(marking_points, i, j):
    angle_threhold = 5
    x_1 = marking_points[i][1]
    y_1 = marking_points[i][2]
    x_2 = marking_points[j][1]
    y_2 = marking_points[j][2]
    for point_idx, tens in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        x_0 = tens[1]
        y_0 = tens[2]
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_1, y_2 - y_1])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        angle = math.acos(np.dot(vec1, vec2)) * (180 / math.pi)
        if angle < angle_threhold:
            return True
    return False


"""## Determine Point Shape
 determine the shape of the slot line based on the vector joining the two slot points and the vector between the point and its direction
"""


def detemine_point_shape(point, vector):
    """Determine which category the point is in."""

    none = 0
    l_down = 2
    t_down = 3
    t_middle = 4
    l_up = 6
    vec_direct = calc_angle_vector(vector)
    # vec_direct_up = 180 - vec_direct
    vec_direct_down = calc_angle_vector([vector[0], -1 * vector[1]])
    marking_direction = calc_angle([point[1], point[2]], [point[3], point[4]])
    if point[5] == 0:
        if abs_direction_diff(vec_direct, marking_direction) > 60 and abs_direction_diff(vec_direct,
                                                                                         marking_direction) < 135:
            return t_middle
        if direction_diff(vec_direct_down, marking_direction) < 50:
            return t_down

    else:
        hz_diff = abs(point[3] - point[1])
        vl_diff = abs(point[4] - point[2])
        if hz_diff > vl_diff:
            difference = marking_direction - vec_direct
            if difference < 170:
                return l_down
            if difference > 170:
                return l_up
        elif hz_diff < vl_diff:
            vl_vector = np.array([0, 500])
            vl_vector = vl_vector / np.linalg.norm(vl_vector)
            marking_vector = np.array([point[3] - point[1], point[4] - point[2]])
            marking_vector = marking_vector / np.linalg.norm(marking_vector)
            dot_product = (np.dot(marking_vector, vl_vector))
            if dot_product > 0.3:
                return l_up
            elif dot_product < 0.3:
                return l_down
    return none


"""## Pair Marking Points to form slots 
according to the shape returned from the "determine_point_shape" function we check whether these two shapes could form a slot or not.
"""


def pair_marking_points(point_a, point_b):
    """See whether two marking points form a slot."""
    point_a, point_b = order_pair2(point_a, point_b)
    vector_ab = np.array([point_b[1] - point_a[1], point_b[2] - point_a[2]])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab)
    point_shape_b = detemine_point_shape(point_b, -vector_ab)
    none = 0
    l_down = 2
    t_down = 3
    t_middle = 4
    l_up = 6

    # shapes :  L up ,   T middle
    #          L up ,   T down
    #          L down , T middle
    #          L down , T down
    #          L down , L down
    #          L down , L up
    #          L up ,L up
    #          T middle  , T middle
    #          T down , T down

    vect1 = np.array([point_a[3] - point_a[1], point_a[4] - point_a[2]])
    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = np.array([point_b[3] - point_b[1], point_b[4] - point_b[2]])
    vect2 = vect2 / np.linalg.norm(vect2)
    dot_product = (np.dot(vect1, vect2))
    if dot_product < -0.5:
        return [0]
    else:

        if point_shape_a == l_up and point_shape_b == l_up:
            return [1, point_shape_a, point_shape_b]
        if point_shape_a == l_down and point_shape_b == l_down:
            return [1, point_shape_a, point_shape_b]
        if point_shape_a == t_middle and point_shape_b == t_middle:
            return [1, point_shape_a, point_shape_b]

        if point_shape_a == t_down and point_shape_b == t_down:
            return [1, point_shape_a, point_shape_b]

        if (point_shape_a == l_up and point_shape_b == t_middle) or (
                point_shape_b == l_up and point_shape_a == t_middle):
            return [1, point_shape_a, point_shape_b]
        if (point_shape_a == t_middle and point_shape_b == l_down) or (
                point_shape_b == t_middle and point_shape_a == l_down):
            return [1, point_shape_a, point_shape_b]

        if (point_shape_a == l_down and point_shape_b == t_down) or (
                point_shape_b == l_down and point_shape_a == t_down):
            return [1, point_shape_a, point_shape_b]
        if (point_shape_a == t_down and point_shape_b == l_up) or (point_shape_b == t_down and point_shape_a == l_up):
            return [1, point_shape_a, point_shape_b]
        if (point_shape_a == l_down and point_shape_b == l_up) or (point_shape_b == l_down and point_shape_a == l_up):
            return [1, point_shape_a, point_shape_b]

    return [0]


"""## Forming Slots

*   the function gets the predicted marking points and then we loop on point by point with the remaining ones and check if they lie between either parallel or perpendicular distance range and then pass it to determine point shape, pair marking point function and based on the returned result it append the slot details (points forming it and its type) to the slot list

"""


def inference_slots(marking_points):
    perpend_min = 120
    perpend_max = 195
    parallel_min = 335
    parallel_max = 370
    num_detected = len(marking_points)#.shape[0]
    perpen_parallel = 0  # perpendicular = 1 , parallel = 2
    slot = {}
    slots = []
    marks_list = []
    slots_list = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            if not (perpend_min <= distance <= perpend_max
                    or parallel_min <= distance <= parallel_max):
                continue

            # Step 2: pass through filtration.
            if pass_through_third_point(marking_points, i, j) and pass_through_third_point(marking_points, j, i):
                continue
            result = pair_marking_points(point_i, point_j)
            if perpend_min <= distance <= perpend_max:
                perpen_parallel = 1  # perpendiculer
            elif parallel_min <= distance <= parallel_max:
                perpen_parallel = 2  # parallel

            if len(result) <= 1:
                if result == 0:
                    continue
            else:
                if result[0] == 1:
                    first_point, second_point = order_pair(marking_points[i], marking_points[j])
                    test1 = {'x1': first_point[1], 'y1': first_point[2], 'dir_x1': first_point[3],
                             'dir_y1': first_point[4],
                             'x2': second_point[1], 'y2': second_point[2], 'dir_x2': second_point[3],
                             'dir_y2': second_point[4],
                             'type': perpen_parallel, 'type1': result[1], 'type2': result[2]}

                    marks_list.append(marking_points[i])
                    marks_list.append(marking_points[j])
                    slots_list.append((i + 1, j + 1, perpen_parallel))
                    slot.update(test1.copy())
                    slots.append(slot.copy())

    slots2 = {'marks': marks_list, 'slots': slots_list}
    return slots, slots2


def order_pair2(point1, point2):
    if abs(point2[2] - point1[2]) < 20:
        if point2[1] > point1[1]:
            return point1, point2
        else:
            return point2, point1
    else:
        if point2[2] > point1[2]:
            return point1, point2
        else:
            return point2, point1


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
            if point1[2] > point2[2]:
                return point1, point2
            else:
                return point2, point1
