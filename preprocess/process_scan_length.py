import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess.preprocess_tools.process_utils import jsonl_save
import csv

"""SCAN SPLIT FROM: https://github.com/i-machine-think/machine-tasks"""

dev_keys = ["length"]
test_keys = ["length"]

train_path = Path('../dataset/scan/length/tasks_train_length.txt')
test_path = {}
test_path["length"] = Path('../dataset/scan/length/tasks_test_length.txt')
dev_path = {}
dev_path["length"] = Path('../dataset/scan/length/tasks_validation_length.txt')

Path('../processed_data/scan_length/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/scan_length/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/scan_length/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/scan_length/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/scan_length/metadata.pkl"))

vocab2count = {}

def process_data(filename, update_vocab=True):
    global vocab2count
    print("\n\nOpening directory: {}\n\n".format(filename))

    srcs = []
    trgs = []

    count = 0

    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            splits = line.split("\t")
            src = splits[0].strip().split(" ") + ["<eos>"]
            trg = splits[1].strip().split(" ") + ["<eos>"]

            srcs.append(src)
            trgs.append(trg)

            if update_vocab:
                for token in src:
                    vocab2count[token] = vocab2count.get(token, 0) + 1
                for token in trg:
                    vocab2count[token] = vocab2count.get(token, 0) + 1

            count += 1

    return srcs, trgs


train_srcs, train_trgs = process_data(train_path, update_vocab=True)

trg_len_max = max([len(trg) for trg in train_trgs])
print("train_len_max: ", trg_len_max)


dev_srcs = {}
dev_trgs = {}
for key in dev_keys:
    print(key)
    dev_srcs[key], dev_trgs[key] = process_data(dev_path[key], update_vocab=False)
    dev_len_max = max([len(src) for src in dev_trgs[key]])
    print("dev_len_max: ", dev_len_max)

"""
train_srcs = [train_srcs[id] for id in train_idx]
train_trgs = [train_trgs[id] for id in train_idx]
"""

test_srcs = {}
test_trgs = {}
for key in test_keys:
    print(key)
    test_srcs[key], test_trgs[key] = process_data(test_path[key], update_vocab=False)
    test_len_max = max([len(src) for src in test_trgs[key]])
    print("test_len_max: ", test_len_max)

counts = []
vocab = []
for word, count in vocab2count.items():
    vocab.append(word)
    counts.append(count)

MAX_VOCAB = 20000
sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]


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
