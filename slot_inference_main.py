import torch
import cv2
from torchvision.transforms import ToTensor
from Slot.slot_inference import inference_slots
from model import init_marking_points_model, image_predict_marking_points
from visualizer import visualize_slot

device = torch.device("cuda")
model = init_marking_points_model()
model = model.eval()

image_path = r"image_path"
input_image = cv2.imread(image_path)
input_image = cv2.medianBlur(input_image, 5)
smoothed = cv2.GaussianBlur(input_image, (9, 9), 10)
input_image = cv2.addWeighted(input_image, 2, smoothed, -1, 5)
image_transform = ToTensor()
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = image_transform(input_image)

predicted_points, mp_image = image_predict_marking_points(input_image, model)

slots, _ = inference_slots(predicted_points)
out_image = visualize_slot(mp_image, slots, True)
