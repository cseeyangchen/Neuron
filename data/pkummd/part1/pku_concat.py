import numpy as np

base_path = '/root/autodl-tmp/FGE/data/pkummd/part1/'

for evaluation in ['xview','xsub']:
    train_datapath = base_path+evaluation+"/train_position64.npy"
    train_labelpath = base_path+evaluation+"/train_label64.npy"
    test_datapath = base_path+evaluation+"/val_position64.npy"
    test_labelpath = base_path+evaluation+"/val_label64.npy"

    train_data = np.load(train_datapath)
    train_label = np.load(train_labelpath)
    test_data = np.load(test_datapath)
    test_label = np.load(test_labelpath)

    data = dict()
    data['x_train'] = train_data
    data['y_train'] = train_label
    data['x_test'] = test_data
    data['y_test'] = test_label

    save_name = base_path+"pkuv1_{}.npz".format(evaluation)

    np.savez(save_name, x_train=train_data, y_train=train_label, x_test=test_data, y_test=test_label)

