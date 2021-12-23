import pickle
import warnings
import argparse
from datetime import datetime
import os 
from typing import Callable, Dict, Iterable
import sys; #sys.path.append('/home/cs19resch11001/FIt-SNE')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, Subset
from torchvision import transforms
#from fast_tsne import fast_tsne


warnings.filterwarnings("ignore")

now = datetime.now()
cur_time = now.strftime("%d-%m-%Y::%H:%M:%S")


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="GEM",
        help="The type (EWC|GEM) of the model architecture",
    )
    args = parser.parse_args(argv)
    return args


args = get_args(sys.argv[1:])

# Creating folder to save weights
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

cache_label = "../data/train_data_x.pth"
# Normal = 0
# Attack = 1
if os.path.exists(cache_label):

    with open("../data/train_data_x.pth", "rb") as f:
        train_data_x = pickle.load(f)
    with open("../data/train_data_y.pth", "rb") as f:
        train_data_y = pickle.load(f)

    with open("../data/test_data_x.pth", "rb") as f:
        test_data_x = pickle.load(f)

    with open("../data/test_data_y.pth", "rb") as f:
        test_data_y = pickle.load(f)
    print("Data Loaded!!!")

else:
    with open("../data/ids_18.pth", "rb") as f:
        df = pickle.load(f)

    y = df.pop(df.columns[-1]).to_frame()

    df["Flow Byts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)
    df["Flow Pkts/s"].replace([np.inf, -np.inf, -np.nan, np.nan], 0, inplace=True)

    for column in df.columns:
        print(
            f"For column -{column}; Max is - {df[column].max()}; Min is {df[column].min()}",
            flush=True,
        )
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min() + 1e-5
        )

    print(
        "Memory usage of normalized_X is : ",
        df.memory_usage().sum() / 1024 ** 2,
        flush=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, stratify=y, test_size=0.33
    )

    del df

    train_dict = {}
    train_label_dict = {}
    test_dict = {}
    test_label_dict = {}

    for i in range(y_train.iloc[:, -1].nunique()):
        train_dict["cat" + str(i)] = X_train[y_train.iloc[:, -1] == i]

        temp = y_train[y_train.iloc[:, -1] == i]

        # Class label 0 = Normal class
        if i == 0:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1

        train_label_dict["cat" + str(i)] = temp

    for i in range(y_test.iloc[:, -1].nunique()):
        test_dict["cat" + str(i)] = X_test[y_test.iloc[:, -1] == i]

        temp = y_test[y_test.iloc[:, -1] == i]

        if i == 0:
            temp.iloc[:, -1] = 0
        else:
            temp.iloc[:, -1] = 1
        test_label_dict["cat" + str(i)] = temp

    train_data_x = list(torch.Tensor(
        train_dict[key].to_numpy()) for key in train_dict)
    train_data_y = list(
        torch.Tensor(train_label_dict[key].to_numpy()) for key in train_label_dict
    )
    test_data_x = list(torch.Tensor(
        test_dict[key].to_numpy()) for key in test_dict)
    test_data_y = list(
        torch.Tensor(test_label_dict[key].to_numpy()) for key in test_label_dict
    )


    with open("../data/train_data_x.pth", "wb") as f:
        pickle.dump(train_data_x, f)

    with open("../data/train_data_y.pth", "wb") as f:
        pickle.dump(train_data_y, f)

    with open("../data/test_data_x.pth", "wb") as f:
        pickle.dump(test_data_x, f)

    with open("../data/test_data_y.pth", "wb") as f:
        pickle.dump(test_data_y, f)
        
whole_x = [t.numpy() for t in train_data_x]
whole_x = np.vstack(whole_x)
whole_x = np.array(whole_x)

whole_y = [t.numpy() for t in train_data_y]
whole_y = np.vstack(whole_y)
whole_y = np.array(whole_y)


# Splitting test samples
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.85, random_state=0)
sss.get_n_splits(whole_x, whole_y)
for train_index, test_index in sss.split(whole_x, whole_y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_t, X_te = whole_x[train_index], whole_x[test_index]
    y_t, y_te = whole_y[train_index], whole_y[test_index]
    break


tsne = TSNE(n_components=2, random_state=0,verbose=1,perplexity=50,init='pca')
X_2d = tsne.fit_transform(X_t)
#X_2d  = fast_tsne(X_t, initialization='random', seed=1, perplexity=50)
target_ids = [0,1]
y_t=y_t.astype('int')

plt.figure(figsize=(10,6))
colors = 'r', 'g'
target_names=['Normal','Attack']
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_2d[y_t[:,-1] == i,0], X_2d[y_t[:,-1] == i,1], c=c, label=label)
plt.axis('off')

plt.legend()
plt.savefig("./tSNE_train_dataset_plot.png")
plt.savefig("./tSNE_train.eps",format='eps')        



       
