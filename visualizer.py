import cv2
import matplotlib.pyplot as plt
import math

from Utils.new_utensils_slotinfrence import calc_angle

"""
# image visulizer
visualize image with labels
"""


def image_visualizer(parking_image):
    plt.imshow(parking_image['image'].permute(1, 2, 0))
    for i in range(len(parking_image['marks'])):
        plt.plot([parking_image['marks'][i][0], parking_image['marks'][i][2]],
                 [parking_image['marks'][i][1], parking_image['marks'][i][3]], 'o')
    plt.show()


"""# image visulizer
visulize image with prediction
"""


def visualize_after_thres(image, prediction, plot=False):

    pre = prediction
    # plt.imshow(image)
    for i in range(len(pre)):
        p0_x = (pre[i, 1].item())
        p0_y = (pre[i][2].item())
        p1_x = pre[i, 3].item()
        p1_y = pre[i][4].item()
        confidence = pre[i][0].item()

        angle = calc_angle([p0_x, p0_y], [p1_x, p1_y]) * (math.pi / 180)
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        p1_x = p0_x + 50 * cos_val
        p1_y = p0_y + 50 * sin_val

        p2_x = p0_x - 50 * sin_val
        p2_y = p0_y + 50 * cos_val

        p3_x = p0_x + 50 * sin_val
        p3_y = p0_y - 50 * cos_val

        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))

        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))

        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))

        p3_x = int(round(p3_x))
        p3_y = int(round(p3_y))

        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv2.putText(image, str(confidence / 100), (p0_x, p0_y),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if pre[i, 5].item() > 50:
            cv2.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv2.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)
    if plot:
        plt.imshow(image)
    return image


"""# slot visulizer
visulize available slot after pairing predicted points
"""


def visualize_slot(image, prediction, plot=False):
    slots = prediction
    # plt.imshow(image.permute(1, 2, 0))
    for i in range(len(slots)):
        x = [slots[i]['x1'].item(), slots[i]['x2'].item()]
        y = [slots[i]['y1'].item(), slots[i]['y2'].item()]
        mark1_x = [slots[i]['x1'].item(), slots[i]['dir_x1'].item()]
        mark1_y = [slots[i]['y1'].item(), slots[i]['dir_y1'].item()]
        mark2_x = [slots[i]['x2'].item(), slots[i]['dir_x2'].item()]
        mark2_y = [slots[i]['y2'].item(), slots[i]['dir_y2'].item()]

        p1 = '1'
        p2 = '2'
        # plt.text(slots[i]['x1']+15, slots[i]['y1']  , p1,fontsize=10 , color='white')
        # plt.text(slots[i]['x2']+15, slots[i]['y2'] , p2,fontsize=10 , color='white')
        txt = 'slot ' + str(i + 1)
        # plt.text(max(slots[i]['x1'],slots[i]['x2']),(slots[i]['y1']+slots[i]['y2']) /2 +15 , txt,fontsize=12 , color='red')
        # dir_x = [slots[i]['dir_x1'],slots[i]['dir_x2']]
        # dir_y = [slots[i]['dir_y1'],slots[i]['dir_y2']]
        # plt.plot(x,y,mark1_x,mark1_y,mark2_x, mark2_y,linewidth=3, markersize=3, color='red')

        cv2.line(image, (int(round(x[0])), int(round(y[0]))), (int(round(x[1])), int(round(y[1]))), (255, 0, 0), 2)
        cv2.line(image, (int(round(mark1_x[0])), int(round(mark1_y[0]))),
                 (int(round(mark1_x[1])), int(round(mark1_y[1]))), (255, 0, 0), 2)
        cv2.line(image, (int(round(mark2_x[0])), int(round(mark2_y[0]))),
                 (int(round(mark2_x[1])), int(round(mark2_y[1]))), (255, 0, 0), 2)
    if plot:
        plt.imshow(image)
    return image
