import torch
import numpy as np

def data_points():
    indices = {
        "80_DCS": [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "25_DCS": [2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22],
        "-20_DCS": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "80_SCS": [2, 3, 4, 5, 6, 7, 8],
        "25_SCS": [9, 10, 11, 12, 13, 14, 15],
        "-20_SCS": [16, 17, 18, 19, 20, 21, 22]}
    return indices

def input_mapping(x, B,mapping_size,data_points):
  if B is None:
    return x
  else:
      B=np.tile(np.expand_dims(B,0),(x.shape[0],1,1))
      x_proj1 = (2.*np.pi*x).reshape(x.shape[0],data_points,2)
      x_proj2 = B.reshape(x.shape[0],2,mapping_size)

      x_proj = x_proj1 @ x_proj2

      return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=2)

def loss_(opt):
    if opt['loss_func'] == 'MSE':
        loss_func = torch.nn.MSELoss(reduction='sum')
    elif opt['loss_func'] == 'L1':
        loss_func = torch.nn.L1Loss()

    return loss_func



