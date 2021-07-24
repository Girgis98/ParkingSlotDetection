from training.marking_points_model import MarkingPointDetector
from Training.loss import my_loss
from Data.dataset import load, collate_mod
import torch_optimizer as optim
from torch.utils.data import DataLoader
import torch
from Training.training import model_training

# main training
device = torch.device("cuda")
dataset_path = r"dataset_path"
park_dataset = load(dataset_path)
data_loader = DataLoader(park_dataset,
                         batch_size = 32, shuffle = True, num_workers = 4, collate_fn = collate_mod, pin_memory = True)
learning_rate = 0.01
num_epochs = 200
batch_size = 1024
model = MarkingPointDetector().to(device)
model = model.to(device)
optimizer = optim.RAdam(model.parameters(), lr = learning_rate)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.9, patience = 300,
                                                        cooldown = 20, min_lr = 0)
scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.0000001, max_lr = 0.01, step_size_up = 50,
                                               step_size_down = 50, mode = 'triangular')
model_training(model, my_loss, optimizer, scheduler1, num_epochs, data_loader, device, batch_size)
