import torch.nn as nn
from Slot.slot_inference import predict, get_predicted_points, non_maximum_suppression
#from model import MarkingPointDetector
from visualizer import visualize_after_thres
import torch
import cv2
from torchvision.transforms import ToTensor

device = torch.device("cpu")


# # Convolutional neural network
# class MarkingPointDetector(nn.Module):
#
#
#     def __init__(self):
#         dropout_prob = 0.0
#         super(MarkingPointDetector, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer91 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer101 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer121 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer131 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer14 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer15 = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer16 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer151 = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer161 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer152 = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer162 = nn.Sequential(
#             nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
#         self.layer17 = nn.Sequential(
#             nn.Conv2d(1024, 6, kernel_size=1, stride=1, padding=0),
#             nn.Identity())
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer91(out)
#         out = self.layer101(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = self.layer121(out)
#         out = self.layer131(out)
#         out = self.layer14(out)
#         out = self.layer15(out)
#         out = self.layer16(out)
#         out = self.layer151(out)
#         out = self.layer161(out)
#         # print(out.shape)
#         out = self.layer152(out)
#         # print(out.shape)
#         out = self.layer162(out)
#         # print(out.shape)
#         out = self.layer17(out)
#         # print(out.shape)
#         out = out.reshape(out.size(0), -1)
#         # print(out.shape)
#
#         return out

# Convolutional neural network
class MarkingPointDetector(nn.Module):


    def __init__(self):
        dropout_prob = 0.0
        super(MarkingPointDetector, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer91 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer101 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer121 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer131 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer14 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer15 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer151 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer161 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer152 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer162 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1), nn.Dropout(p=dropout_prob))
        self.layer17 = nn.Sequential(
            nn.Conv2d(1024, 6, kernel_size=1, stride=1, padding=0),
            nn.Identity())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer91(out)
        out = self.layer101(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer121(out)
        out = self.layer131(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer151(out)
        out = self.layer161(out)
        # print(out.shape)
        out = self.layer152(out)
        # print(out.shape)
        out = self.layer162(out)
        # print(out.shape)
        out = self.layer17(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)

        return out

"""
### initialize marking points predictor
"""


def init_marking_points_model():
    """
    :return: model:  model after initialization with its weights
    """
    model = MarkingPointDetector()#.to(device)
    checkpoint = torch.load(r'350k_after_shape_cp1_v10',map_location=torch.device('cpu'))  # weights path
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


"""### Takes an image returns predicted Marking points"""


def image_predict_marking_points(input_image, model):
    """
    :param input_image: input image to be passed for model for prediction
    :param model: model used for prediction
    :return: predict2: predicted points after applying confidence threshold and non max suppression
    :return: out_image: output image from visualizer after drawing predicted points
    """
    image_transform = ToTensor()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = image_transform(input_image)

    model = model.eval()
    predict_awal = predict(input_image.reshape((1, 3, 512, 512)),model,device)#.to(device))
    predict_ba3d = get_predicted_points(predict_awal, 70)
    predict2 = non_maximum_suppression(predict_ba3d)
    out_image = visualize_after_thres(input_image, predict2[0], True)
    return predict2, out_image
