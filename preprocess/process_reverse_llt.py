import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess.preprocess_tools.process_utils import jsonl_save
import csv

dev_keys = ["normal"]
test_keys = ["normal", "long1", "long2", "long3", "long4", "long5"]

train_path = Path('LongLookupTables/LongLookupTablesReverse/data/train.tsv')

dev_path = {}
dev_path["normal"] = Path('LongLookupTables/LongLookupTablesReverse/data/validation.tsv')

test_path = {}
test_path["normal"] = Path('LongLookupTables/LongLookupTablesReverse/data/heldout_inputs.tsv')
test_path["long1"] = Path('LongLookupTables/LongLookupTablesReverse/data/longer_seen_1.tsv')
test_path["long2"] = Path('LongLookupTables/LongLookupTablesReverse/data/longer_seen_2.tsv')
test_path["long3"] = Path('LongLookupTables/LongLookupTablesReverse/data/longer_seen_3.tsv')
test_path["long4"] = Path('LongLookupTables/LongLookupTablesReverse/data/longer_seen_4.tsv')
test_path["long5"] = Path('LongLookupTables/LongLookupTablesReverse/data/longer_seen_5.tsv')

Path('../processed_data/rllt/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/rllt/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/rllt/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/rllt/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/rllt/metadata.pkl"))

vocab2count = {}


def process_data(filename, update_vocab=True):
    global vocab2count
    print("\n\nOpening directory: {}\n\n".format(filename))

    srcs = []
    trgs = []

    count = 0

    with open(filename) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for row in tsv_file:
            src = row[0].strip().split(" ")
            trg = row[1].strip().split(" ") + ["<eos>"]
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

dev_srcs = {}
dev_trgs = {}
for key in dev_keys:
    dev_srcs[key], dev_trgs[key] = process_data(dev_path[key], update_vocab=False)

test_srcs = {}
test_trgs = {}
for key in test_keys:
    test_srcs[key], test_trgs[key] = process_data(test_path[key], update_vocab=False)

idx = [i for i in range(len(test_srcs["long1"]))]
random.shuffle(idx)
dev_idx = idx[0:500]
test_idx = idx[500:]

for id in dev_idx:
    dev_srcs["normal"].append(test_srcs["long1"][id])
    dev_trgs["normal"].append(test_trgs["long1"][id])


test_srcs["long1"] = [test_srcs["long1"][id] for id in test_idx]
test_trgs["long1"] = [test_trgs["long1"][id] for id in test_idx]


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
