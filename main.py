import io
import pandas as pd
import numpy as np

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
  return ids, descs, seqs


def main():
    pos_file_path = "T6SE_training_data/T6SE_Training_Pos_138.fasta"
    neg_file_path = "T6SE_training_data/T6SE_Training_Neg_1112.fasta"
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
    seqs = np.array(list(map(str, seqs)), dtype=str)

    # DataFrame

    names = ["id", "desc", "seq", "is_positive"]
    data = pd.DataFrame(data=np.c_[ids, descs, seqs, isPositive], columns=names)

    X_train, X_test, y_train, y_test = train_test_split(seqs, isPositive, test_size=0.33, random_state=0)

    model = MultiModel()
    
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

if __name__ == '__main__':
    main()