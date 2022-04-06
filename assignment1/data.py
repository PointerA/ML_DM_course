from cProfile import label
import pandas as pd
import numpy as np

# Make types to labels dictionary
def type_to_label_dict(all_type):
    # input -- types
    # output -- type_to_label dictionary
    type_to_label_dict = {}
    for i in range(len(all_type)):
        type_to_label_dict[all_type[i]] = i
    return type_to_label_dict



# Convert types to labels
def convert_type_to_label(types, type_to_label_dict):
    # input -- list of types, and type_to_label dictionary
    # output -- list of labels
    types = types.tolist()
    labels = list()
    for t in types:
        labels.append(type_to_label_dict[t[0]])
    return labels

#normalize
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean)/std
    return data

def readData(file, seed=7, proportion=0.7):
    df=pd.read_excel(file)
    r,c = df.shape
    # random choice train set and test set
    types = list(set(df["Class"]))
    types.sort()
    train_x, test_x = np.empty((0,c-1)), np.empty((0,c-1))
    train_y, test_y = np.empty((0,1)), np.empty((0,1))
    for t in types:
        per_df = df[df['Class'] == t]
        train = per_df.sample(frac=proportion,random_state=seed,axis=0)
        train_x = np.vstack((train_x,train.drop('Class', axis=1).values))
        train_y = np.vstack((train_y, train["Class"].values.reshape(-1,1)))
        test = per_df[~per_df.index.isin(train.index)]
        test_x = np.vstack((test_x,test.drop('Class', axis=1).values))
        test_y = np.vstack((test_y, test["Class"].values.reshape(-1,1)))

    
    dict = type_to_label_dict(types)
    train_y = convert_type_to_label(train_y, dict)
    test_y = convert_type_to_label(test_y, dict)
    train_x = normalize(train_x)
    test_x = normalize(test_x)
    return train_x, test_x, train_y, test_y, dict
