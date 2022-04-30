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

def readData(file):
    if file[-4:] == "xlsx":
        df=pd.read_excel(file)
    elif file[-3:] == "csv":
        df=pd.read_csv(file)
    types = list(set(df["Class"]))
    types.sort()
    dict = type_to_label_dict(types)
    data_x = df.drop('Class', axis=1).values
    data_y = df["Class"].values.reshape(-1,1)
    data_y = convert_type_to_label(data_y, dict)
    data_x = normalize(data_x)
    return data_x, data_y, len(types)
