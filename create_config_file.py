import yaml

users = [{'configuration':{'config':"DCS", 'temp': 80},'Fourier':True, 'gpu': 1,'checkpoint': '80_DCS_10bar_sorted_fourier_mapping',
          'dataset':{'sensor_range': [-1,10], 'data_path':'datasets/10_bar.pkl','test_sensors': 100,'shuffle':False},\
          'logger':{'print_freq': 10000,'save_checkpoint_freq': 10000},\
          'training_settings':{'batch_size': 32, 'lr':0.0085,'epochs': 1500000, 'temp_epochs':3000, 'scheduler': 'MultiStepLR','gamma':0.5, 'weight_80':1.0, 'weight_25':1.0,\
                               'weight_-20':1.0},\
           'factor':40,
          'train': True, 'resume_train': False,
           'GT_min':-0.0239, 'GT_max': 0.255, 'p_min':-0.38, 'p_max':0.38, 'loss_func': 'MSE'}]

with open('configs/80_DCS_10bar_fourier_mapping.yml', 'w') as f:
    data = yaml.dump(users, f, sort_keys=False)
