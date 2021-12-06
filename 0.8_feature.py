import numpy as np
import pandas as pd

f = open('select_data.txt', 'r')

data = f.read()
data = data.split(' ')

dataset = np.loadtxt('colon_cancer_data.csv', delimiter=",", skiprows=1)
dataset_attr = np.loadtxt('colon_cancer_data.csv', delimiter=",", dtype=str)
dataset_attr = dataset_attr[0]

label = dataset[:, -1]
label_attr = dataset_attr[-1]
new_dataset = []
new_dataset_attr = []

for idx in enumerate(data):
    if idx[1] == '1':
        if len(new_dataset) == 0:
            new_dataset = dataset[:, idx[0]]
            new_dataset = np.expand_dims(new_dataset, axis=1)
            new_dataset_attr.append(dataset_attr[idx[0]])
        else:
            data = np.expand_dims(dataset[:, idx[0]], axis=1)
            new_dataset = np.concatenate((new_dataset, data), axis=1)
            new_dataset_attr.append(dataset_attr[idx[0]])

label = np.expand_dims(label, axis=1)
new_dataset_attr.append(label_attr)
new_dataset = np.concatenate((new_dataset, label), axis=1)

new_dataset_attr = np.expand_dims(new_dataset_attr, axis=0)
new_dataset = np.concatenate((new_dataset_attr, new_dataset), axis=0)

df = pd.DataFrame(new_dataset)
df.to_csv('0.80_feature.csv', header=False, index=False, encoding='utf-8')


