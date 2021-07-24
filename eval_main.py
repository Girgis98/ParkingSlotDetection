from Utils.process import complete_marking_vector_label_mult
from model import init_marking_points_model
from Data.dataset import load, collate_mod
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda")
testing_dataset_path = r"dataset_path"
park_dataset = load(testing_dataset_path)
data_loader = DataLoader(park_dataset,
                         batch_size = 32, shuffle = True, num_workers = 4, collate_fn = collate_mod, pin_memory = True)

model = init_marking_points_model()
model = model.eval()

accum_f1 = 0
accum_accu = 0
accum_prec = 0
accum_rec = 0
ctr = 0
with torch.no_grad():
    for i_batch, train_batch in enumerate(data_loader):
        images = train_batch['image'].to(device)
        labels = complete_marking_vector_label_mult(train_batch).to(device)
        outputs = model(images).to(device)
        outputs = outputs.reshape((-1, 6, 16, 16))
        accuracy, evaluation, precision, recall = eval(outputs.float(), labels.float())
        accum_f1 += evaluation
        accum_accu += accuracy
        accum_prec += precision
        accum_rec += recall
        ctr += 1
        print("Current F1 score is:", (accum_f1 / ctr) * 100, "\nCurrent Accuracy is:", (accum_accu / ctr) * 100,
              "\nCurrent Precision is:", (accum_prec / ctr) * 100, "\nCurrent Recall is:", (accum_rec / ctr) * 100,
              "\n\n")
print("Final F1 score is:", (accum_f1 / ctr) * 100, "\nFinal Accuracy is:", (accum_accu / ctr) * 100,
      "\nFinal Precision is:", (accum_prec / ctr) * 100, "\nFinal Recall is:", (accum_rec / ctr) * 100, "\n\n")
