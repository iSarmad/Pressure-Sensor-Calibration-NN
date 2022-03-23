import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler as min_max
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.nn.init import kaiming_normal_, constant_

def save_checkpoint(state, iters,model):
    filename = model + "_" + str(iters) + '.pth.tar'
    torch.save(state, filename)

def scaler(data, scaler, axis=1):
    # demonstrate data normalization with sklearn
    # fit scaler on data
    if axis == 1:
        data = np.expand_dims(data, axis)
    scaler.fit(data)
    # apply transform
    l = scaler.data_max_

    normalized = scaler.transform(data)
    normalized = np.squeeze(normalized)

    # inverse transform
    # inverse = scaler.inverse_transform(normalized)
    return normalized, scaler

def temp_train( x, x1, y1,lr,batch_size, iters,path):
    plt.ion()
    net = Temp_Net()

    net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    loss_func = torch.nn.MSELoss()

    ep = 0

    train_data = CustomDataset(x1, y1, x)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print("Training temperature NN first")
    for t in range(ep, iters+1):
        epoch_loss = []

        for i, batch in enumerate(train_loader):
            net.train()
            x_temp2 = batch[0]
            y_temp1 = batch[1]
            x_temp1 = batch[2]


            x_temp2 = x_temp2.cuda()
            x_temp1 = x_temp1.cuda()
            y_temp1 = y_temp1.cuda()

            prediction, coeff = net(x_temp1, x_temp2)

            loss = loss_func(torch.squeeze(prediction), torch.squeeze(y_temp1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)

        if t % 10 == 0:
            loss1 = sum(epoch_loss) / len(epoch_loss)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Steps: ", t + 1, "Loss MSE: ",
                  loss1.cpu().detach().numpy(),
                  'Learning Rate: ', lr)

        if t % 1000 == 0:
            save_checkpoint(net.state_dict(), t,path)

def poly_temp(xin, x):
    with torch.no_grad():
        out = x[:, 0].unsqueeze(1) * xin[:, 0, :] ** 1 + \
              x[:, 1].unsqueeze(1)
        out = out.unsqueeze(1)

    return out


def temp_test(index,x_test,model_path):
    """
        test(index, sensors,model_path)

        Finds out-of-bound and within-bound sensors and plots them for 80 , 25 and -20 c cycle
    """
    net = Temp_Net()
    net.cuda()

    net.load_state_dict(torch.load(model_path))

    x = x_test.cuda()

    net.eval()

    x_inp = x[:, :, index]

    with torch.no_grad():
        pred, coeff = net(x_inp, x)

    prediction = poly_temp(x, coeff)

    return prediction.cpu().detach().numpy()

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, x_partial):
        self.x = x_tensor
        self.x_partial = x_partial
        self.y = y_tensor


    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.x_partial[index])

    def __len__(self):
        return len(self.x)

class Temp_Net(torch.nn.Module):
    def __init__(self):
        super(Temp_Net, self).__init__()

        self.conv1 = torch.nn.Conv1d(1,64, 1)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)
        self.relu = torch.nn.ReLU()


        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, xin):

        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)

        out = x[:, 0].unsqueeze(1) * xin[:, 0, :] ** 1 + x[:, 1].unsqueeze(1)

        return out.unsqueeze(1), x;

