import collections
import string
from typing import Dict, List, Tuple
import pickle
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
from absl import logging
import tensorflow_datasets as tfds

dev_keys = ["cfq"]
test_keys = ["length"]


Path('../processed_data/cfq_length/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/cfq_length/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/cfq_length/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/cfq_length/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/cfq_length/metadata.pkl"))

Dataset = Dict[str, List[Tuple[str, str]]]


def tokenize_punctuation(text):
  text = map(lambda c: ' %s ' % c if c in string.punctuation else c, text)
  return ' '.join(''.join(text).split())


def preprocess_sparql(query):
  """Do various preprocessing on the SPARQL query."""
  # Tokenize braces.
  query = query.replace('count(*)', 'count ( * )')

  tokens = []
  for token in query.split():
    # Replace 'ns:' prefixes.
    if token.startswith('ns:'):
      token = token[3:]
    # Replace mid prefixes.
    if token.startswith('m.'):
      token = 'm_' + token[2:]
    tokens.append(token)

  return ' '.join(tokens).replace('\\n', ' ')


def get_encode_decode_pair(sample):
  # Apply some simple preprocessing on the tokenizaton, which improves the
  # performance of the models significantly.
  encode_text = tokenize_punctuation(sample['questionPatternModEntities'])
  decode_text = preprocess_sparql(sample['sparqlPatternModEntities'])
  return (encode_text, decode_text)


def get_dataset_from_tfds(dataset, split):
  """Load dataset from TFDS and do some basic preprocessing."""
  logging.info('Loading dataset via TFDS.')
  allsplits = tfds.load(dataset + '/' + split, as_supervised=True)
  if 'validation' in allsplits:
    # CFQ and divergence splits of StarCFQ have all three sets.
    split_names = {'train': 'train', 'dev': 'validation', 'test': 'test'}
  else:
    # Scan and non-divergence splits of StarCFQ have 'train' and 'test' sets
    # only. We simply output the test set as both dev and test. We only really
    # use the dev set but t2t-datagen expects all three.
    split_names = {'train': 'train', 'dev': 'test', 'test': 'test'}

  dataset = collections.defaultdict(list)
  for cfq_split_name, tfds_split_name in split_names.items():
    for raw_x, raw_y in tfds.as_numpy(allsplits[tfds_split_name]):
      encode_decode_pair = (tokenize_punctuation(raw_x.decode()),
                            preprocess_sparql(raw_y.decode()))
      dataset[cfq_split_name].append(encode_decode_pair)

  size_str = ', '.join(f'{s}={len(dataset[s])}' for s in split_names)
  logging.info('Finished loading splits. Size: %s', size_str)
  return dataset

datasets = get_dataset_from_tfds("cfq", split="query_complexity_split")


train_srcs = [t[0] for t in datasets["train"]]
train_trgs = [t[1] for t in datasets["train"]]

dev_srcs = [t[0] for t in datasets["dev"]]
dev_trgs = [t[1] for t in datasets["dev"]]

test_srcs = [t[0] for t in datasets["test"]]
test_trgs = [t[1] for t in datasets["test"]]

vocab2count = {}
train_srcs_ = []
train_trgs_ = []

for x, y in zip(train_srcs, train_trgs):
  x = x.split(" ") + ["<eos>"]
  y = y.split(" ") + ["<eos>"]
  for token in x:
    vocab2count[token] = vocab2count.get(token, 0) + 1
  for token in y:
    vocab2count[token] = vocab2count.get(token, 0) + 1
  train_srcs_.append(x)
  train_trgs_.append(y)

train_srcs = train_srcs_
train_trgs = train_trgs_

dev_srcs_ = []
dev_trgs_ = []
for x, y in zip(dev_srcs, dev_trgs):
  x = x.split(" ") + ["<eos>"]
  y = y.split(" ") + ["<eos>"]
  dev_srcs_.append(x)
  dev_trgs_.append(y)

dev_srcs = {}
dev_trgs = {}
dev_srcs[dev_keys[0]] = dev_srcs_
dev_trgs[dev_keys[0]] = dev_trgs_


test_srcs_ = []
test_trgs_ = []
for x, y in zip(test_srcs, test_trgs):
  x = x.split(" ") + ["<eos>"]
  y = y.split(" ") + ["<eos>"]
  test_srcs_.append(x)
  test_trgs_.append(y)

test_srcs = {}
test_trgs = {}
test_srcs[test_keys[0]] = test_srcs_
test_trgs[test_keys[0]] = test_trgs_


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



