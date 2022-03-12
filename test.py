import io
import pandas as pd
import numpy as np
from collections import OrderedDict

from Bio import SeqIO
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, confusion_matrix

from featurescalculator import  FeaturesCalculator
from multimodel import  MultiModel

def read_file_into_lists(handle : io.TextIOWrapper) -> tuple:
  ids, descs, seqs = [], [], []
  for record in SeqIO.parse(handle, "fasta"):
        ids.append(record.id)
        descs.append(record.description)
        seqs.append(record.seq)
  str_seq = list(map(str, seqs))
  return ids, descs, str_seq

def read_data_for_training() -> pd.DataFrame:
    pos_file_path = "data/T6SE_training_data/T6SE_Training_Pos_138.fasta"
    neg_file_path = "data/T6SE_training_data/T6SE_Training_Neg_1112.fasta"
    ids, descs, seqs, isPositive = [], [], [], []

    # считывание

    with open(pos_file_path, mode='r', encoding='utf8') as handle:
        temp_ids, temp_descs, temp_seqs = read_file_into_lists(handle)
        ids += temp_ids
        descs += temp_descs
        seqs += temp_seqs
        isPositive += [True] * len(temp_ids)
    with open(neg_file_path, mode='r', encoding='utf8') as handle:
        temp_ids, temp_descs, temp_seqs = read_file_into_lists(handle)
        ids += temp_ids
        descs += temp_descs
        seqs += temp_seqs
        isPositive += [False] * len(temp_ids)
    del temp_ids, temp_descs, temp_seqs # очистка памяти

    # обёртка

    ids, descs = np.array(ids, dtype=str), np.array(descs)
    isPositive = np.array(isPositive, dtype=bool)

    # DataFrame

    names = ["id", "desc", "seq", "is_positive"]
    data = pd.DataFrame(data=np.c_[ids, descs, seqs, isPositive], columns=names)
    return data

def read_data_for_compare() -> pd.DataFrame:
    feature_file_path = "data/for_compare/ids_and_features.csv"
    seq_file_path = "data/for_compare/ids_and_seqs.faa"
    with open(seq_file_path, mode='r', encoding='utf-8') as handle:
        ids, descs, seqs = read_file_into_lists(handle)
    temp = OrderedDict([(i, [seq]) for i, seq in zip(ids,seqs)])
    del ids, descs, seqs
    temp_df = pd.read_csv(feature_file_path)
    n = len(temp_df)
    for i in range(n):
        row = temp_df.iloc[i]
        name = row["Name"][1::]
        aac, dpc, qso, t6se = row["AAC"], row["DPC"], row["QSO"], str(row["T6SE"] == "Yes")
        temp[name].extend([aac, dpc, qso, t6se])
    rows = [[key,*temp[key]] for key in temp.keys()]
    del temp
    data = pd.DataFrame(columns=("id", "seq", "AAC", "DPC", "QSO", "is_positive"))
    for i in range(n):
        data.loc[i] = rows[i]
    return data


def main():
    data = read_data_for_training()
    X, y = data["seq"].to_numpy(), data["is_positive"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    calculator = FeaturesCalculator()

    model = MultiModel(features = calculator.features)
    
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    
    data = read_data_for_compare()
    X_comp = data["seq"].to_numpy()
    y_true = data["is_positive"].to_numpy()
    print(model.score(X_comp, y_true))

    compare_vecs = np.array([[i, j, k] for i, j, k in zip(data["AAC"].to_numpy(), data["DPC"].to_numpy(), data["QSO"].to_numpy())])
    computed_vecs = model.transform(X_comp)
    features = ["AAC", "DPC", "QSO"]
    temp = model.transform(X_comp)
    computed_vecs = np.array([[i, j, k] for i, j, k in zip(temp[:,0], temp[:,1], temp[:,2])])
    n = len(compare_vecs)
    X_plot = list(range(n))
    
    for i in range(3):
        print("MSE",features[i], mean_squared_error(compare_vecs[i], computed_vecs[i]))
    
    for i in range(3):
        plt.title("MSE " + features[i])
        plt.plot(X_plot, compare_vecs[:,i], "r-",marker="o", label="bastion")
        plt.plot(X_plot, computed_vecs[:,i],"b-",marker="o", label="our work")
        plt.legend()
        plt.show()

    y_pred = model.predict(X_comp) == "True"
    y_true = y_true == "True"
    fpr, tpr, _ = roc_curve(y_true,  y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.title('ROC-AUC')
    plt.plot(fpr,tpr,label = f"AUC = {auc}")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 10)
    plt.show()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    plt.title('Prediction rate')
    labels = ("True negative", "False positive", "False negative", "True positive")
    explode = (0.1, 0.1, 0.1, 0.1)
    temp = np.array([tn, fp, fn, tp])/len(y_pred) * 100
    plt.pie(temp, labels=labels, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.show()


if __name__ == '__main__':
    main()