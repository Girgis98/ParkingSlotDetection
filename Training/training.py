import math
import torch
from torch.utils.tensorboard import SummaryWriter
from Utils.process import complete_marking_vector_label_mult

"""## Training"""


def model_training(model, myloss, optimizer, scheduler, num_epochs, data_loader, device, actual_batch_size = 1024):
    graph_itrator = 0
    graph_itrator_2 = 0
    batch_loss = 0
    writer = SummaryWriter('save_path')  # for tensorboard visualization
    BS = data_loader.batch_size
    actual_batch_factor = actual_batch_size / BS
    number_of_batches = math.ceil(len(data_loader.dataset.sample_names) / BS)
    model = model.train()

    for epoch in range(num_epochs):
        for i_batch, train_batch in enumerate(data_loader):
            images = train_batch['image'].to(device)
            labels = complete_marking_vector_label_mult(train_batch).to(device)

            # Forward pass
            outputs = model(images).to(device)
            outputs = outputs.reshape((-1, 6, 16, 16))
            myloss = myloss(outputs.float(), labels.float())

            # Backward and optimize
            myloss.backward()

            batch_loss += myloss.item()

            if graph_itrator % actual_batch_factor == 0:
                optimizer.step()
                optimizer.zero_grad()
                if graph_itrator > 10:
                    scheduler.step(batch_loss)
                    s = scheduler

                    for i in range(len(optimizer.param_groups)):
                        lrr = s.optimizer.param_groups[i]['lr']
                    for g in optimizer.param_groups:
                        g['lr'] = lrr

            for g in optimizer.param_groups:
                lr = g['lr']

            if graph_itrator % 120 == 0:
                torch.save({
                    'i_batch': i_batch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': myloss,
                    'scheduler': scheduler,
                    'batch_loss': batch_loss
                }, r'weights_save_path')

            if graph_itrator % actual_batch_factor == 0:
                graph_itrator_2 += 1
                writer.add_scalar('training loss',
                                  batch_loss / actual_batch_factor,
                                  graph_itrator_2)
                writer.add_scalar('lr',
                                  lr,
                                  graph_itrator_2)
                batch_loss = 0
                graph_itrator += 1
    return 0

