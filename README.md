# Pressure-Sensor-Calibration-using-Neural-Networks
Official Code for AAAI 2022 Paper : Reducing Energy Consumption of Pressure Sensor Calibration Using Polynomial HyperNetworks with Fourier Features
https://ojs.aaai.org/index.php/AAAI/article/view/21474
To install package in a conda environemnt

conda install --file requirements.txt


Run create_config_file.py to first create a config file. Specify all necessary parameters.

configuration

config:
DCS: Double Cycle Selection
SCS: Single Cycle Selection

temp:
80, 25, -20

For example, 80 DCS would mean the used temperature cycles to be 25C and -20C and 80 SCS would mean used temp cycle to be 80C.

checkpoint name should be of this order: [temp]_[config (DCS/SCS)]_[sensor_range]_[sorted/shuffled]_[mapping or no mapping]

You can create your own checkpoints by adding additional info in checkpoint name.

Weights of the temperature cycles can be adjusted by changing parameters weight_80, weight_25 and weight_-20.

GT_min, GT_max are the values close to the min max range of the GT of the dataset.

If Fourier is true, run B_gauss.py to create a frequency matrix first.

Run train.py by specifying the relevant config file to train the model.

Run test.py to test the model. Specify the model path.
