import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler as min_max
import os
import pickle
from temperature_prescaling import temp_train,temp_test
random.seed(42)

class Create_Dataset():
    def __init__(self, opt, index):

        self.sensor_range = opt["dataset"]["sensor_range"]
        self.index = index
        self.GT_min = opt["GT_min"]
        self.GT_max = opt["GT_max"]
        self.pscaled_min = opt["p_min"]
        self.pscaled_max = opt["p_max"]
        self.data_path = opt["dataset"]["data_path"]
        self.n_test = opt["dataset"]["test_sensors"]
        self.shuffle = opt["dataset"]["shuffle"]
        self.temp_epochs = opt['training_settings']['temp_epochs']
        self.batch_size = opt['training_settings']['batch_size']
        self.lr = opt['training_settings']['lr']

        self.temp_folder = "_".join(opt['checkpoint'].split("_")[0:4])

        self.factor = opt['factor']
        if self.sensor_range == [-1,10]:

            data_path = "datasets/10_bar.pkl"
        elif self.sensor_range == [-1,40]:
            data_path = "datasets/40_bar.pkl"

        with open(data_path, 'rb') as fp:
            self.nn_data = pickle.load(fp)


    def scaler(self,data, scaler, axis=1):
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

    def poly_temp(self,xin, x):
        with torch.no_grad():
            out = x[:, 0].unsqueeze(1) * xin[:, 0, :] ** 1 + \
                  x[:, 1].unsqueeze(1)
            out = out.unsqueeze(1)

        return out

    def lstsq_compensation(self,poly_pow, raw_data, gt):
        """
                lstsq_compensation(poly_pow, data, gt_index, gt_size)

                Calculates polynomial coefficients with torch's least square algorithm.

                Input:

                poly_pow: Matrix of exponents for building a polynomial.
                data: Numpy array - First part is ground truth, second part is asic.
                gt_index: Index of ground truth as input for least square algorithm.
                gt_size: First columns of data array is gt. Set index to the first measurement of the asic.

                Return:

                numpy array as polynomial coefficients in order and dimension of poly_pow

            """
        # Build polynomials without coefficients:
        poly = np.zeros((raw_data.shape[0], poly_pow.shape[0]), dtype=np.float64)

        for i in range(raw_data.shape[0]):
            temp = np.power(raw_data[i], poly_pow)
            poly[i] = np.cumprod(temp, 1)[:, -1]

        # Calculate polynomial coefficients:
        c = torch.from_numpy(gt).unfold(0, 1, 1)
        R, _ = torch.lstsq(c, torch.from_numpy(poly))

        return R[0:poly.shape[1]].numpy()

    def normalization_P(self, X,min,max):

        min_val = self.pscaled_min
        max_val = self.pscaled_max
        X_std = (X - min_val) / (max_val - min_val)
        X_scaled = X_std * (max - min) + min

        return X_scaled, min_val, max_val

    def normalization_T(self,X,min,max):

        min_val = np.min(X)
        max_val = np.max(X)
        X_std = (X - min_val) / (max_val - min_val)
        X_scaled = X_std * (max - min) + min

        return X_scaled, min_val, max_val

    def normalization_GT(self,X,min,max):

        min_val = self.GT_min
        max_val = self.GT_max
        X_std = (X - min_val) / (max_val - min_val)
        X_scaled = X_std * (max - min) + min

        return X_scaled, min_val, max_val

    def denormalization(self,X_std,min_val,max_val, factor):

        X = X_std * (max_val - min_val) + min_val
        X = X*factor

        return X

    def normalization_test(self,X, min,max, min1,max1):

        X_std = (X - min) / (max - min)
        X_scaled = X_std * (max1 - min1) + min1

        return X_scaled

    def pressure_prescale(self,p_np, y_np):
        index = self.index

        p_np_train = p_np[index, :]

        y_np = y_np[index, :] * self.factor

        x = torch.tensor([[self.sensor_range[0]], [self.sensor_range[1]]], dtype=torch.float64, requires_grad=False)
        y = torch.tensor([[-0.35], [0.35]], dtype=torch.float64, requires_grad=False)

        x_1 = torch.cat((torch.ones(x.size(0), 1), x), 1)

        R, _ = torch.lstsq(y, x_1)

        transform_prescale_dig_1_linear = R[0:x_1.size(1)].numpy()[::-1]

        for i in range(y_np.shape[1]):
            y_np[:, i] = 1. * transform_prescale_dig_1_linear[1][0] + y_np[:, i] * transform_prescale_dig_1_linear[0][0]

        p_np_train = p_np_train.astype(np.float64)
        p_np = p_np.astype(np.float64)
        y_np = y_np.astype(np.float64)
        c = []
        poly_pow = np.zeros((2, 4))
        poly_pow[0][0] = 1

        for i in range(y_np.shape[1]):
            prescale_1_coeff = self.lstsq_compensation(poly_pow, p_np_train[:, i], y_np[:, i])
            c.append(prescale_1_coeff)

        c = np.array(c)

        pressure2 = np.zeros((23, p_np.shape[1]))

        for i in range(p_np.shape[1]):
            a = c[i, 0]
            b = c[i, 1]

            pressure2[:, i] = a[0] * p_np[:, i] ** 1 + b[0]

        return pressure2

    def temperature_prescale(self,t_raw):


        t_raw_train = t_raw[:, self.train_sensors]

        t_prescaled, scaler_t = self.scaler(t_raw_train, min_max(feature_range=(-0.35, 0.35)), axis=2)

        t_raw_test = t_raw[:, self.test_sensors]

        t = np.expand_dims(t_raw_train, axis=1)
        t = np.transpose(t, (2, 1, 0))

        t = torch.from_numpy(t)
        x = t.type(torch.FloatTensor)

        t = np.expand_dims(t_raw, axis=1)
        t = np.transpose(t, (2, 1, 0))

        t = torch.from_numpy(t)
        t_raw = t.type(torch.FloatTensor)

        t_raw_test = np.expand_dims(t_raw_test, axis=1)
        t_raw_test = np.transpose(t_raw_test, (2, 1, 0))

        t_raw_test = torch.from_numpy(t_raw_test)
        x_test = t_raw_test.type(torch.FloatTensor)

        t_prescaled = np.expand_dims(t_prescaled, axis=1)
        t_prescaled = np.transpose(t_prescaled, (2, 1, 0))

        t_prescaled = torch.from_numpy(t_prescaled)
        y = t_prescaled.type(torch.FloatTensor)

        x_inp = x[:, :, self.index]
        temp_model_folder = 'checkpoints/temperature/'+self.temp_folder

        if not os.path.exists(temp_model_folder ):
            os.makedirs(temp_model_folder )

        temp_model_path = temp_model_folder + "/" + self.temp_folder +"_" + str(self.temp_epochs) + '.pth.tar'

        if not os.path.exists(temp_model_path):
            temp_train(x_inp, x, y, self.lr,self.batch_size, iters=self.temp_epochs, path=temp_model_folder)
            print("Training of temperature NN complete")
            print("###################################")

        print("Loading temperature trained model")
        print("###################################")

        temp_prescaled = temp_test(self.index, t_raw, temp_model_path)

        return temp_prescaled

    def data(self, norm=True):

        if not self.shuffle:
            self.test_sensors = list(range(self.nn_data['raw_p'].shape[0] - self.n_test, self.nn_data['raw_p'].shape[0]))

        else:
            self.test_sensors = sorted(random.sample(range(0, self.nn_data['raw_p'].shape[0]), self.n_test))

        self.train_sensors = [x for x in list(range(self.nn_data['raw_p'].shape[0])) if x not in self.test_sensors]

        nn_data = self.nn_data

        p_raw = np.transpose(nn_data['raw_p'], (1, 0))
        t_raw = np.transpose(nn_data['raw_t'], (1, 0))
        gt = np.transpose(nn_data['gt'], (1, 0))
        poly = np.transpose(nn_data['poly'], (1, 0))

        p_prescaled = self.pressure_prescale(p_raw, gt)

        t_prescaled = self.temperature_prescale(t_raw)
        t_prescaled = np.transpose(t_prescaled[:, 0, :], (1, 0))

        y_np = gt[:, self.train_sensors]
        y_np_test = gt[:, self.test_sensors]

        p_np = p_prescaled[:, self.train_sensors]
        p_np_test = p_prescaled[:, self.test_sensors]

        t_np = t_prescaled[:, self.train_sensors]
        t_np_test = t_prescaled[:, self.test_sensors]

        poly_np = poly[:, self.train_sensors]
        poly_np_test = poly[:, self.test_sensors]

        if norm is True:
            # Normalizing data

            print("Prescaled Pressure Range Train", "[", np.min(p_np), ",", np.max(p_np), "]")

            p_np, min_p, max_p = self.normalization_P(p_np, -1, 1)
            print("Normalized Pressure Range Train", "[", np.min(p_np), ",", np.max(p_np), "]")
            print("******************************")

            print("Prescaled Pressure Range Test", "[", np.min(p_np_test), ",", np.max(p_np_test), "]")
            p_np_test = self.normalization_test(p_np_test, min_p, max_p, -1, 1)
            print("Normalized Pressure Range Test", "[", np.min(p_np_test), ",", np.max(p_np_test), "]")
            print("******************************")

            print("Prescaled Temperature Range Train", "[", np.min(t_np), ",", np.max(t_np), "]")

            t_np, min_t, max_t = self.normalization_T(t_np, -1, 1)
            print("Normalized Temperature Range Train", "[", np.min(t_np), ",", np.max(t_np), "]")
            print("******************************")

            print("Prescaled Temperature Range Test", "[", np.min(t_np_test), ",", np.max(t_np_test), "]")
            t_np_test = self.normalization_test(t_np_test, min_t, max_t, -1, 1)
            print("Normalized Temperature Range Test", "[", np.min(t_np_test), ",", np.max(t_np_test), "]")
            print("******************************")

            print("Raw GT Range Train", "[", np.min(y_np), ",", np.max(y_np), "]")
            y_np, min, max = self.normalization_GT(y_np, 0, 1)
            print("Normalized GT Range Train", "[", np.min(y_np), ",", np.max(y_np), "]")
            print("******************************")

            print("Raw GT Range Test", "[", np.min(y_np_test), ",", np.max(y_np_test), "]")
            y_np_test = self.normalization_test(y_np_test, min, max, 0, 1)
            print("Normalized GT Range Test", "[", np.min(y_np_test), ",", np.max(y_np_test), "]")
            print("******************************")

            poly_np_test = self.normalization_test(poly_np_test, min, max, 0, 1)
            poly_np = self.normalization_test(poly_np, min, max, 0, 1)


        else:
            poly_np = np.expand_dims(poly_np, 1)

            y_np_test = np.expand_dims(y_np_test, 1)  # Normalizing Polynomial data for comparison
            poly_np_test = np.expand_dims(poly_np_test, 1)  # Normalizing Polynomial data for comparison

        x_np = np.asarray([p_np, t_np])
        x_np_test = np.asarray([p_np_test, t_np_test])

        x_np = np.transpose(x_np, (2, 0, 1))

        x_t = torch.from_numpy(x_np)
        x = x_t.type(torch.FloatTensor)

        x_np_test = np.transpose(x_np_test, (2, 0, 1))

        x_t_test = torch.from_numpy(x_np_test)
        x_test = x_t_test.type(torch.FloatTensor)

        y_np = np.transpose(y_np, (1, 0))
        y_np = np.expand_dims(y_np, axis=2)

        y_t = torch.from_numpy(y_np)
        y = y_t.type(torch.FloatTensor)

        y_np_test = np.transpose(y_np_test, (1, 0))
        y_np_test = np.expand_dims(y_np_test, axis=2)

        y_t_test = torch.from_numpy(y_np_test)
        y_test = y_t_test.type(torch.FloatTensor)

        poly_np = np.transpose(poly_np, (1, 0))
        poly_np = np.expand_dims(poly_np, axis=2)

        poly_np_test = np.transpose(poly_np_test, (1, 0))
        poly_np_test = np.expand_dims(poly_np_test, axis=2)

        return x, y, poly_np, x_test, y_test, poly_np_test


if __name__ == "__main__":

    Data = Create_Dataset()
    x, y,poly_np, x_test, y_test, poly_np_test = Data.data()


