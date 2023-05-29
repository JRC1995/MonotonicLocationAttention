import torch as T
import numpy as np
import random
import copy

class seq2seq_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train
        self.UNK_id = config["UNK_id"]
        self.vocab_len = config["vocab_len"]
        self.eos_id = config["vocab2idx"]["<eos>"]
        self.vocab2idx = config["vocab2idx"]

    def pad(self, items, PAD):
        # max_len = max([len(item) for item in items])
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks


    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]

    def collate_fn(self, batch):
        batch = copy.deepcopy(batch)


        srcs = [obj['src'] for obj in batch]
        trgs = [obj['trg'] for obj in batch]

        srcs_vec = [obj['src_vec'] for obj in batch]
        ptr_srcs_vec = [obj["ptr_src_vec"] for obj in batch]
        trgs_vec = [obj['trg_vec'] for obj in batch]
        if self.config["pointer"]:
            labels = [obj['label'] for obj in batch]
        else:
            labels = trgs_vec

        if self.config["no_eos"]:
            trgs_vec =[x[:-1] for x in trgs_vec]
            labels = [x[:-1] for x in labels]
        oov_nums = [obj["oov_num"] for obj in batch]
        max_oov_num = max(oov_nums)

        bucket_size = len(srcs)
        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        lengths = [len(src) + len(trg) for src, trg in zip(srcs, trgs)]
        sorted_idx = np.argsort(lengths)

        srcs = self.sort_list(srcs, sorted_idx)
        trgs = self.sort_list(trgs, sorted_idx)
        srcs_vec = self.sort_list(srcs_vec, sorted_idx)
        ptr_srcs_vec = self.sort_list(ptr_srcs_vec, sorted_idx)
        trgs_vec = self.sort_list(trgs_vec, sorted_idx)
        labels = self.sort_list(labels, sorted_idx)


        meta_batches = []
        i = 0
        while i < bucket_size:
            batches = []

            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            inr_ = inr

            j = copy.deepcopy(i)
            while j < i + inr:
                srcs_vec_, src_masks = self.pad(srcs_vec[j:j + inr_], PAD=self.PAD)
                trgs_vec_, trg_masks = self.pad(trgs_vec[j:j + inr_], PAD=self.PAD)
                labels_, _ = self.pad(labels[j:j + inr_], PAD=self.PAD)
                ptr_srcs_vec_, _ = self.pad(ptr_srcs_vec[j:j + inr_], PAD=self.PAD)

                batch = {}
                batch["src_vec"] = T.tensor(srcs_vec_).long()
                trg_vec = T.tensor(trgs_vec_).long()
                batch["trg_vec"] = trg_vec
                batch["ptr_src_vec"] = T.tensor(ptr_srcs_vec_).long()
                batch["src"] = srcs[j:j + inr_]
                batch["trg"] = trgs[j:j + inr_]
                batch["src_mask"] = T.tensor(src_masks).float()
                batch["trg_mask"] = T.tensor(trg_masks).float()
                labels_ = T.tensor(labels_).long()
                assert labels_.size() == trg_vec.size()
                batch["labels"] = labels_
                batch["batch_size"] = inr_
                batch["max_oov_num"] = max_oov_num
                batches.append(batch)
                j += inr_
            i += inr
            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
