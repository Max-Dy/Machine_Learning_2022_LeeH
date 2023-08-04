from HW1 import My_Model, predict, torch
from HW1 import x_train, device, config, test_loader
import csv


def save_pred(preds, file):
    with open(file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'preds13.csv')
