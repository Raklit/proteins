import io
import pandas as pd
import numpy as np
from collections import OrderedDict

from Bio import SeqIO
import graphviz

from sklearn.model_selection import train_test_split

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

    model = MultiModel()
    
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    
    data = read_data_for_compare()
    X_comp = data["seq"].to_numpy()
    y_true = data["is_positive"].to_numpy()
    print(model.score(X_comp, y_true))

if __name__ == '__main__':
    main()