import argparse
import torch
from model import Net
from torch.utils.data import Dataset, DataLoader
from data import Create_Dataset
import os
import yaml
from utils import data_points, loss_, input_mapping
import numpy as np
import matplotlib.pyplot as plt
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str,
                        default='configs/80_DCS_10bar_fourier_mapping.yml',
                        help='Path to option YAML file')
    parser.add_argument('-test_model', type=str,
                        default='checkpoints/compensation/80_DCS_10bar_sorted_fourier_mapping/0.pth.tar',
                        help='Path to option YAML file')

    args = parser.parse_args()
    return args


def out_of_bound_vs_iterations(index, dataset, folder, model,xin_test, xin_train):
    """
        out_of_bound_vs_iterations(index, sensors, train)
        Plots no. of out of bound files against training iterations for 80 , 25 and -20 c cycle
        inputs

        index: The indices of cycle removed in case of double cycle selection or the retained cycle in case single cycle selection
        sensors: Sensors to plot results on
        train: True: Training sensors, False: Testing sensors

    """
    x_train, y_train, poly_np_train, x_test, y_test, poly_np_test = dataset

    ep = 0
    n80 = []
    n25 = []
    n20 = []

    n80_test = []
    n25_test = []
    n20_test = []

    x_axis = []

    print("Out-of-bound files for test and train set")

    for file in os.listdir(folder):

            p = folder + "/" + str(ep) + '.pth.tar'

            model.load_state_dict(torch.load(p))

            n_test = inference(x_test, y_test,model,xin_test)
            n = inference(x_train, y_train,model,xin_train)

            print("Epoch ", ep)
            print("Test ", n_test)
            print("Train ", n)
            print("*********************")


            n80.append(n[0])
            n25.append(n[1])
            n20.append(n[2])


            n80_test.append(n_test[0])
            n25_test.append(n_test[1])
            n20_test.append(n_test[2])

            x_axis.append(ep)
            ep += 10000


    fig1, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(24,20))

    line5, = ax1.plot(x_axis[:], n80[:], 'bo--', linewidth=2.0, label='80C')
    line6, = ax1.plot(x_axis[:], n25[:], 'g.-', linewidth=2.0, label='25C')
    line7, = ax1.plot(x_axis[:], n20[:], 'r.-', linewidth=2.0, label='-20C')

    ax1.grid()  # line1.set_antialiased(False)
    ax1.set_ylabel('#OB', fontsize=37)
    ax1.set_xlabel('Epochs', fontsize=37)
    ax1.legend(prop={"size": 37})
    ax1.set_ylim(0,500)
    ax1.set_xlim(0, x_axis[-1])

    ax1.xaxis.get_offset_text().set_fontsize(24)

    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)

    ff = 'No. of out-of-bound training sensors (Total sensors ' + str(x_train.shape[0]) + ')'

    ax1.set_title(ff, fontsize=40)
    plt.subplots_adjust(hspace=0.4)

    line5, = ax2.plot(x_axis[:], n80_test[:], 'bo--', linewidth=2.0, label='80C')
    line6, = ax2.plot(x_axis[:], n25_test[:], 'g.-', linewidth=2.0, label='25C')
    line7, = ax2.plot(x_axis[:], n20_test[:], 'r.-', linewidth=2.0, label='-20C')

    ax2.grid()
    ax2.set_ylabel('#OB', fontsize=37)
    ax2.set_xlabel('Epochs', fontsize=37)
    ax2.legend(prop={"size": 37})
    ax2.set_ylim(0, 50)
    ax2.set_xlim(0, x_axis[-1])
    ax2.xaxis.get_offset_text().set_fontsize(24)

    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)

    ff = 'No. of out-of-bound testing sensors (Total sensors' + str(x_test.shape[0]) + ')'

    ax2.set_title(ff, fontsize=40)

    path1 = 'plot.jpg'
    plt.savefig(path1)
    plt.pause(0.01)
    plt.close('all')

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

def inference(x, y, model,x_inp):

    x = x.cuda()
    y = y.cuda()
    x_inp = x_inp.cuda()
    model.eval()

    temp1 = np.repeat(0.15, 2)
    temp2 = np.repeat(0.2, 7)
    temp3 = np.repeat(0.15, 7)
    temp4 = np.repeat(0.2, 7)

    err_bound_train = np.concatenate((temp1, temp2, temp3, temp4), axis=0)
    err_bound_train = np.expand_dims(err_bound_train, axis=0)
    err_bound_train = np.repeat(err_bound_train, np.shape(x)[0], axis=0)
    err_bound_train = np.expand_dims(err_bound_train, axis=2)

    test_cyc = []

    test_cyc1 = list(range(2, 9))  # 80c
    test_cyc2 = list(range(0, 2)) + list(range(9, 16))  # 25c
    test_cyc3 = list(range(16, 23))

    with torch.no_grad():
        prediction, coeff = model(x_inp, x)

    test_cyc.append([test_cyc1, test_cyc2, test_cyc3])

    out_bound_files = []

    for i in range(3):

        test = test_cyc[0][i]

        err_bound_test = err_bound_train[:, test, :]

        x_test = x[:, :, test]

        y_test = y[:, test, :]

        prediction_test = poly_nn(x_test, coeff)

        # Testing on all dataset
        y_axis = []

        ts = x.shape[0]

        for i in range(ts):
            y_npp = y_test[i, :, :].cpu().detach().numpy()
            pred_npp = prediction_test[i, :, :].cpu().detach().numpy()
            pred_npp = np.transpose(pred_npp, (1, 0))
            err_plt_test_pos1 = err_bound_test[i, :, :] * 0.01
            err_plt_test_neg1 = -err_bound_test[i, :, :] * 0.01
            error = y_npp - pred_npp

            out_bound = []
            out_bound1 = []
            err = []
            for j in range(len(test)):
                if error[j] > err_plt_test_pos1[j] or error[j] < err_plt_test_neg1[j]:

                    out_bound.append(abs(error[j]))
                    out_bound1.append(abs(error[j]))
                    err.append(err_plt_test_pos1[j])
                else:
                    out_bound.append(0)

            is_all_zero = np.all((np.array(out_bound) == 0))

            if is_all_zero:
                y_axis.append(0)
            else:

                diff = np.divide(np.subtract(np.array(out_bound1), np.array(err)), np.array(out_bound1))
                diff = list(diff)
                max_bb = max(diff)
                # Appending percentage difference if out of bound
                y_axis.append(max_bb)

        zeros = []
        ones = []

        for i in range(ts):
            if y_axis[i] == 0:
                zeros.append(i)
            else:
                ones.append(i)

        y_axis2 = [y_axis[i] for i in ones]

        out_bound_files.append(len(y_axis2))


    return out_bound_files

def envelope(x_in,dataset,model, temp_used):
    x_train, y_train, poly_np_train, x_test, y_test, poly_np_test = dataset

    x = x_test.cuda()
    y = y_test.cuda()

    x_in = x_in.cuda()

    model.eval()

    temp1 = np.repeat(0.15, 2)
    temp2 = np.repeat(0.2, 7)
    temp3 = np.repeat(0.15, 7)
    temp4 = np.repeat(0.2, 7)

    err_bound_train = np.concatenate((temp1, temp2, temp3, temp4), axis=0)
    err_bound_train = np.expand_dims(err_bound_train, axis=0)
    err_bound_train = np.repeat(err_bound_train, np.shape(x)[0], axis=0)
    err_bound_train = np.expand_dims(err_bound_train, axis=2)

    prediction, coeff = model(x_in, x)

    prediction_test1 = poly_nn(x, coeff)

    j = -1
    #plt.figure(1,figsize=(24, 20))
    fig1,  ax3 = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    MSE = []

    for file in range(x_test.shape[0]):

        j += 1

        y_np2 = y[file, :, :].cpu().detach().numpy()
        pred_np = prediction_test1[file, :, :].cpu().detach().numpy()
        pred_np = np.transpose(pred_np, (1, 0))

        x_test_temp = x[file, :, :].cpu().detach().numpy()

        steps = np.arange(1, np.size(x_test_temp, axis=1) + 1, 1)
        err_plt_test = err_bound_train[file, :, :] * 0.01

        line1, = ax3.plot(steps, y_np2 - pred_np, 'r.-', linewidth=2.0,
                          label='$p_{GT}$ - $p_{c}$' if j == 0 else "")
        line3, = ax3.plot(steps, err_plt_test, 'k.-', linewidth=2.0, label='Error_Bound' if j == 0 else "")
        line4, = ax3.plot(steps, -err_plt_test, 'k.-', linewidth=2.0, label='Error_Bound' if j == 0 else "")
        # line5, = plt.plot(steps, y_np2 - pt, 'y', linewidth=2.0, label='Polynomial Fit error' if j == 0 else "");

        #line1.set_antialiased(False)
        ax3.set_ylabel('Error',fontsize=37)
        ax3.set_xlabel('Setpoint',fontsize=37)
        #ax3.legend(loc=(0.98, 0))

        ax3.legend(prop={"size": 20},loc=(0.68,0.8))
        ax3.text(0.06, 0.75, '25C',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax3.transAxes,
                color='black', fontsize=20)
        ax3.text(0.27, 0.75, '80C',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax3.transAxes,
                 color='black', fontsize=20)

        ax3.text(0.55, 0.75, '25C',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax3.transAxes,
                 color='black', fontsize=20)

        ax3.text(0.87, 0.75, '-20C',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax3.transAxes,
                 color='black', fontsize=20)


        ax3.tick_params(axis='x', labelsize=20)
        ax3.tick_params(axis='y', labelsize=20)

        ax3.set_title('Temp Cycles Used :' + temp_used, fontsize=40)
        xint = range(min(steps), math.ceil(max(steps)) + 1)
        #plt.tight_layout()
        plt.xticks(xint)

        mse = ((y_np2 - pred_np) ** 2).mean(axis=0)
        MSE.append(mse)

    mse = np.mean(MSE)
    #ax2.grid()
    ax3.grid()
    path = "Figures"
    plt.vlines(x=[2, 9, 16, 23], ymin=[-0.006, -0.006, -0.006,-0.006], ymax=[0.006, 0.006, 0.006,0.006], colors='teal', ls='--', lw=5,
               label='vline_multiple - partial height')

    if not os.path.exists(path):
        os.makedirs(path)

    path1 = path + "/Error_Band.png"
    plt.savefig(path1)
    plt.pause(0.01)

def main():
    args = get_args()

    torch.backends.cudnn.benchmark = True

    with open(args.opt) as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)

    dict = dict[0]


    if dict['Fourier']:

        model = Net(16*2)
    else:

        model = Net(2)

    index = data_points()[str(dict["configuration"]["temp"]) + "_" + dict["configuration"]["config"]]
    dataset = Create_Dataset(dict, index).data(norm=True)

    if dict["configuration"]["config"] == "DCS":
        if dict["configuration"]["temp"] == 80:
            temp_used = '-20C, 25C'
        if dict["configuration"]["temp"] == -20:
            temp_used = '80C, 25C'
        if dict["configuration"]["temp"] == 25:
            temp_used = '-20C, 80C'

    elif dict["configuration"]["config"] == "SCS":
        temp_used = dict["configuration"]["temp"]+"C"

    x_train, y_train, poly_np_train, x_test, y_test, poly_np_test = dataset

    xin_test = x_test[:, :, index]
    xin_train = x_train[:, :, index]

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

        x_in = input_mapping(xin_test, B_dict['gauss_1.0'], mapping_size=16, data_points=len(index))
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[2], x_in.shape[1])
        a = torch.from_numpy(x_in)
        xin_test = a.type(torch.FloatTensor)

        x_in = input_mapping(xin_train, B_dict['gauss_1.0'], mapping_size=16, data_points=len(index))
        x_in = x_in.reshape(x_in.shape[0], x_in.shape[2], x_in.shape[1])
        a = torch.from_numpy(x_in)
        xin_train = a.type(torch.FloatTensor)

    model = model.cuda()

    test_model = args.test_model

    model.load_state_dict(torch.load(test_model))

    envelope(xin_test,dataset,model, temp_used)

    folder = 'checkpoints/compensation/'+dict['checkpoint']

    out_of_bound_vs_iterations(index,dataset,folder,model,xin_test,xin_train)


if __name__ == "__main__":
    main()
