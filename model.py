import torch.nn as nn
from Slot.slot_inference import predict, get_predicted_points, non_maximum_suppression
from model import MarkingPointDetector
from visualizer import visualize_after_thres
import torch
import cv2
from torchvision.transforms import ToTensor

device = torch.device("cuda")


# Convolutional neural network
class MarkingPointDetector(nn.Module):

    def __init__(self):
        super(MarkingPointDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 128, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1))

        self.predict = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 6, kernel_size = 1, stride = 1, padding = 0)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.predict(out)
        out = out.reshape(out.size(0), -1)
        return out


"""
### initialize marking points predictor
"""


def init_marking_points_model():
    model = MarkingPointDetector().to(device)
    checkpoint = torch.load(r'/model_weights')  # weights path
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


"""### Takes an image returns predicted Marking points"""


def image_predict_marking_points(input_image, model):
    image_transform = ToTensor()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = image_transform(input_image)

    model = model.eval()
    predict_awal = predict(input_image.reshape((1, 3, 512, 512)).to(device))
    predict_ba3d = get_predicted_points(predict_awal, 70)
    predict2 = non_maximum_suppression(predict_ba3d)
    out_image = visualize_after_thres(input_image, predict2[0], False)
    return predict2, out_image
