import torch
from torch.nn.init import kaiming_normal_, constant_
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,map_size):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv1d(map_size,64, 1)

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)
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

        return out.unsqueeze(1), x

