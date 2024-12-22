from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
from wrs import rm, mcm, wd

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)

nupdate = 100
trail_num = 100
seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = 'SVM' 


if __name__ == '__main__':
    '''save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{backbone}_seed_selection_large'
    save_path = f'0000_test_programs/nn_ik/results/{rootpath}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    '''data processing'''
    file = 'seed_selection_larger_dataset'
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/{file}.npz')
    y, query_point, dist2query, delta_q, seed_jnt_value = dataset['label'], dataset['query_point'], dataset['dist2query'], dataset['delta_q'], dataset['seed_jnt_value']
    x = np.concatenate([delta_q, seed_jnt_value], axis=1)
    # x = np.concatenate([query_point, dist2query, delta_q, seed_jnt_value], axis=1)

    '''normalization and data loader'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    x_uniform_train = scaler.fit_transform(x_train)
    x_uniform_test = scaler.transform(x_test)

    '''model'''
    if backbone == 'LR':
        model = LogisticRegression(class_weight='balanced', random_state=42)
    elif backbone == 'SVM':
        # model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        model = SVC(kernel='rbf', probability=True, random_state=42)

    model.fit(x_uniform_train, y_train)
    y_pred = model.predict(x_uniform_test)

    if backbone == 'LR':
        y_pred_proba = model.predict_proba(x_uniform_test)[:, 1]
    elif backbone == 'SVM':
        y_pred_proba = model.predict_proba(x_uniform_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    model_path = os.path.join(save_path, f'{backbone.lower()}_model.pkl')
    scaler_path = os.path.join(save_path, 'scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved at: {model_path}")
    print(f"Scaler saved at: {scaler_path}")
