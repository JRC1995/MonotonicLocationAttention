import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess.preprocess_tools.process_utils import jsonl_save
import csv

dev_keys = ["normal"]
test_keys = ["IID", "long15", "long20", "long25", "long30", "long50", "long100"]

key2lens = {"IID": [5, 10],
            "long15": [15, 15],
            "long20": [20, 20],
            "long25": [25, 25],
            "long30": [30, 30],
            "long50": [50, 50],
            "long100": [100, 100]}


vocab = [str(i) for i in range(10)]

def num2repeat(num):
    if num <= 3:
        return 1
    elif num <= 6:
        return 3
    else:
        return 5


def create_data(min_length, max_length, max_samples=10000):
    global vocab
    global repeat_dict

    total_samples = {}
    while len(total_samples) <= max_samples:
        if min_length == max_length:
            length = max_length
        else:
            choices = [i for i in range(min_length, max_length + 1)]
            length = random.choice(choices)
        X = []
        Y = []
        for i in range(length):
            token = random.choice(vocab)
            X.append(token)
            Y = [token] + Y
        total_samples[" ".join(X)] = Y + ["<eos>"]

    srcs = []
    trgs = []
    for X, Y in total_samples.items():
        srcs.append(X.split(" "))
        trgs.append(Y)
        print("src: ", X)
        print("trg: ", Y)

    return srcs, trgs


Path('../processed_data/rc/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/rc/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/rc/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/rc/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/rc/metadata.pkl"))

vocab2count = {}

train_srcs, train_trgs = create_data(min_length=5, max_length=10)

dev_srcs = {}
dev_trgs = {}
for key in dev_keys:
    dev_srcs[key], dev_trgs[key] = create_data(min_length=10, max_length=15, max_samples=2000)

test_srcs = {}
test_trgs = {}
for key in test_keys:
    test_srcs[key], test_trgs[key] = create_data(min_length=key2lens[key][0],
                                                 max_length=key2lens[key][1],
                                                 max_samples=2000)


vocab2idx = {}
for i, token in enumerate(vocab):
    vocab2idx[token] = i

special_tags = ["<sos>", "<unk>", "<eos>", "<pad>"]
for token in special_tags:
    if token not in vocab2idx:
        vocab2idx[token] = len(vocab2idx)


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<unk>']) for word in text]


def create_ptr_src(srcs, srcs_vec):
    ptr_srcs_vec = []
    oov_nums = []

    global vocab2idx
    vocab_len = len(vocab2idx)
    UNK_id = vocab2idx["<unk>"]

    for src, src_vec in zip(srcs, srcs_vec):
        ptr_src_vec = []
        src_token_dict = {}
        for token in src:
            if token not in src_token_dict and token not in vocab2idx:
                src_token_dict[token] = vocab_len + len(src_token_dict)
        for token, token_id in zip(src, src_vec):
            if token_id == UNK_id:
                ptr_src_vec.append(src_token_dict.get(token, UNK_id))
            else:
                ptr_src_vec.append(token_id)
        ptr_srcs_vec.append(ptr_src_vec)
        oov_nums.append(len(src_token_dict))
    return ptr_srcs_vec, oov_nums


def create_labels(trg, trg_vec, src):
    global vocab2idx
    vocab_len = len(vocab2idx)
    UNK_id = vocab2idx["<unk>"]
    label = []
    src_token_dict = {}
    for token in src:
        if token not in src_token_dict and token not in vocab2idx:
            src_token_dict[token] = vocab_len + len(src_token_dict)
    for token, id in zip(trg, trg_vec):
        if (id == UNK_id) and (token in src_token_dict):
            label.append(src_token_dict.get(token, UNK_id))
        else:
            label.append(id)

    assert len(label) == len(trg_vec)

    return label


def vectorize_data(srcs, trgs):
    data_dict = {}
    srcs_vec = [text_vectorize(src) for src in srcs]
    trgs_vec = [text_vectorize(trg) for trg in trgs]
    ptr_srcs_vec, oov_nums = create_ptr_src(srcs, srcs_vec)
    labels = [create_labels(trg, trg_vec, src) for trg_vec, src, trg in zip(trgs_vec, srcs, trgs)]

    data_dict["src"] = srcs
    data_dict["trg"] = trgs

    data_dict["src_vec"] = srcs_vec
    data_dict["ptr_src_vec"] = ptr_srcs_vec
    data_dict["oov_num"] = oov_nums

    data_dict["trg_vec"] = trgs_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_srcs, train_trgs)
jsonl_save(filepath=train_save_path,
           data_dict=train_data)

dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_srcs[key], dev_trgs[key])
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_srcs[key], test_trgs[key])
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

metadata = {"dev_keys": dev_keys,
            "test_keys": test_keys,
            "vocab2idx": vocab2idx}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
