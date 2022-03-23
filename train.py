import argparse
import torch
from model import Net
from torch.utils.data import Dataset, DataLoader
from data import Create_Dataset
import os
import yaml
from utils import data_points, loss_, input_mapping
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str,
                        default='configs/80_DCS_10bar_fourier_mapping.yml',
                        help='Path to option YAML file')

    args = parser.parse_args()
    return args


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, x_partial):
        self.x = x_tensor
        self.x_partial = x_partial
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.x_partial[index])

    def __len__(self):
        return len(self.x)

def poly_nn(xin, x):
    with torch.no_grad():
        out = x[:, 0].unsqueeze(1) * xin[:, 0, :] ** 4 + \
              x[:, 1].unsqueeze(1) * xin[:, 0, :] ** 3 + \
              x[:, 2].unsqueeze(1) * xin[:, 0, :] ** 2 * xin[:, 1, :] ** 1 + \
              x[:, 3].unsqueeze(1) * xin[:, 0, :] ** 2 + \
              x[:, 4].unsqueeze(1) * xin[:, 0, :] ** 1 * xin[:, 1, :] ** 2 + \
              x[:, 5].unsqueeze(1) * xin[:, 0, :] ** 1 * xin[:, 1, :] ** 1 + \
              x[:, 6].unsqueeze(1) * xin[:, 0, :] ** 1 + \
              x[:, 7].unsqueeze(1) * xin[:, 1, :] ** 2 + \
              x[:, 8].unsqueeze(1) * xin[:, 1, :] ** 1 + \
              x[:, 9].unsqueeze(1)
        out = out.unsqueeze(1)
    return out
def save_checkpoint(state, iters, opt):
    model_folder = 'checkpoints/compensation/' + opt["checkpoint"]

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_path = model_folder + "/" + str(iters) + '.pth.tar'
    torch.save(state, model_path)


def main():
    args = get_args()

    torch.backends.cudnn.benchmark = True

    with open(args.opt) as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)

    dict = dict[0]

    if dict['resume_train']:
        path = dict['pretrained_model_path']
        resume_state = torch.load(path)
    else:
        resume_state = None

    loss_func = loss_(dict)

    if dict['Fourier']:

        model = Net(16*2)
    else:

        model = Net(2)

    if resume_state:
        resume_epoch = int(path.split(("_"))[-1].split(("."))[0])
        model.load_state_dict(resume_state)

    else:
        resume_epoch = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=dict['training_settings']['lr'])
    index = data_points()[str(dict["configuration"]["temp"]) + "_" + dict["configuration"]["config"]]
    dataset = Create_Dataset(dict, index).data(norm=True)

    x_train, y_train, poly_np_train, x_test, y_test, poly_np_test = dataset

    x_in = x_train[:, :, index]

    if dict['Fourier'] == True:
        B_dict = {}
        # Standard network - no mapping
        B_dict['none'] = None
        # Basic mapping
        B_dict['basic'] = np.eye(2)

        try:
            B_gauss = np.load("B_gauss.npy")
        except:
            raise Exception('First run B_gauss.py to save the numpy file')

        # Three different scales of Gaussian Fourier feature mappings
        for scale in [1., 10., 100.]:
            B_dict[f'gauss_{scale}'] = B_gauss * scale

        x_in = input_mapping(x_in, B_dict['gauss_1.0'], mapping_size=16, data_points=len(index))
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[2], x_in.shape[1])

    batch_size = dict['training_settings']['batch_size']
    train_data = CustomDataset(x_train, y_train, x_in)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=dict['dataset']['shuffle'])

    model = model.cuda()

    folder = 'checkpoints/compensation/'+ dict["checkpoint"]

    if not os.path.exists(folder):
        os.makedirs(folder)

    w80 = dict['training_settings']['weight_80']
    w25 = dict['training_settings']['weight_25']
    w20 = dict['training_settings']['weight_-20']

    for t in range(resume_epoch, dict['training_settings']["epochs"]):
        epoch_loss = []

        for i, batch in enumerate(train_loader):
            model.train()
            x_temp2 = batch[0]
            y_temp1 = batch[1]
            x_temp1 = batch[2]

            x_temp2 = x_temp2.cuda()
            x_temp1 = x_temp1.cuda()
            y_temp1 = y_temp1.cuda()

            prediction, coeff = model(x_temp1, x_temp2)

            prediction1 = prediction[:, :, 0:2]
            prediction2 = prediction[:, :, 2:9]
            prediction3 = prediction[:, :, 9:16]
            prediction4 = prediction[:, :, 16:23]

            ytemp1 = y_temp1[:, 0:2, :]
            ytemp2 = y_temp1[:, 2:9, :]
            ytemp3 = y_temp1[:, 9:16, :]
            ytemp4 = y_temp1[:, 16:23, :]

            loss1 = 1.0 * loss_func(torch.squeeze(prediction1), torch.squeeze(ytemp1))
            loss2 = w80 * loss_func(torch.squeeze(prediction2), torch.squeeze(ytemp2))
            loss3 = w25 * loss_func(torch.squeeze(prediction3), torch.squeeze(ytemp3))
            loss4 = w20 * loss_func(torch.squeeze(prediction4), torch.squeeze(ytemp4))

            loss = (loss1 + loss2 + loss3 + loss4) / (23 * batch_size)

            # loss = loss_func(torch.squeeze(prediction), torch.squeeze(y_temp1))

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            epoch_loss.append(loss)

        if t % dict['logger']['print_freq'] == 0:
            # plot and show learning process
            loss1 = sum(epoch_loss) / len(epoch_loss)
            for param_group in optimizer.param_groups:
                lr = dict['training_settings']['lr']
            print("Steps: ", t + 1, "Loss MSE: ",
                  loss1.cpu().detach().numpy(),
                  'Learning Rate: ', lr, 'Polynomial Loss MSE:',
                  ((y_train.cpu().detach().numpy() - poly_np_train) ** 2).mean(axis=0)[0])

        # save the model
        if t % dict['logger']["save_checkpoint_freq"] == 0:
            save_checkpoint(model.state_dict(), t, dict)


if __name__ == '__main__':
    main()





