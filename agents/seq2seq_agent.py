import torch as T
import torch.nn as nn
from torch.optim import *
import copy
import math
import torch.nn.functional as F
import numpy as np
import nltk

class NoamLRSched:
    def __init__(self, lr: float, state_size: int, warmup_steps: int):
        self.lr = lr / (state_size ** 0.5)
        self.warmup_steps = warmup_steps

    def get(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.lr / float(step + 1) ** 0.5
        else:
            return self.lr / (self.warmup_steps ** 1.5) * float(step + 1)


class seq2seq_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = eval(config["optimizer"])
        self.pad_id = config["PAD_id"]

        if self.config["custom_betas"]:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"],
                                       betas=(0.9, 0.98))
        else:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"])

        self.key = "none"
        # self.label_smoothing = self.config["label_smoothing"]
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.vocab2idx = config["vocab2idx"]
        self.idx2vocab = {id: token for token, id in self.vocab2idx.items()}
        self.vocab_len = len(config["vocab2idx"])
        self.global_step = 0
        self.eps = 1e-9
        if self.config["schedule"]:
            self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='max',
                                                                    factor=config["scheduler_reduce_factor"],
                                                                    patience=config["scheduler_patience"])
        else:
            self.scheduler = None
        self.epoch_level_scheduler = True

    def loss_fn(self, logits, labels, output_mask, penalty_item=None):
        vocab_len = logits.size(-1)
        N, S = labels.size()
        assert logits.size() == (N, S, vocab_len)
        assert output_mask.size() == (N, S)

        true_dist = F.one_hot(labels, num_classes=vocab_len)
        assert true_dist.size() == (N, S, vocab_len)

        neg_log_logits = -T.log(logits + self.eps)

        assert true_dist.size() == neg_log_logits.size()

        cross_entropy = T.sum(neg_log_logits * true_dist, dim=-1)

        assert cross_entropy.size() == (N, S)
        masked_cross_entropy = cross_entropy * output_mask
        loss = T.sum(masked_cross_entropy) / T.sum(output_mask)

        if penalty_item is not None:
            loss = loss + penalty_item

        return loss

    def decode(self, prediction_idx, src):
        src_token_dict = {}
        for token in src:
            if token not in src_token_dict and token not in self.vocab2idx:
                src_token_dict[token] = self.vocab_len + len(src_token_dict)

        src_token_dict_rev = {v: k for k, v in src_token_dict.items()}

        decoded_seq = []
        for id in prediction_idx:
            if id >= self.vocab_len:
                word = src_token_dict_rev[id]
            else:
                word = self.idx2vocab[id]
            if word == "<eos>":
                break
            decoded_seq.append(word)

        return decoded_seq

    def evaluate(self, preds, golds):
        matches = 0
        total = 0
        ed = 0
        for pred, gold in zip(preds, golds):
            gold = " ".join(gold).split("<eos>")[0].strip().split(" ")
            if self.config["no_eos"]:
                pred = pred[0:len(gold)]
            else:
                pred = " ".join(pred).split("<eos>")[0].strip().split(" ")
                if self.config["cheat_eos"]:
                    gold = gold[0:len(pred)]

            if "edit_measure" in self.config and self.config["edit_measure"]:
                ed = ed + nltk.edit_distance(pred, gold)

            gold = " ".join(gold).strip()
            pred = " ".join(pred).strip()
            if pred == gold:
                matches += 1
            total += 1

        accuracy = 0 if total == 0 else matches / total

        return {"accuracy": accuracy * 100, "ed": ed, "total": total, "correct_predictions": matches}

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        if not self.DataParallel:
            batch["src_vec"] = batch["src_vec"].to(self.device)
            batch["trg_vec"] = batch["trg_vec"].to(self.device)
            batch["ptr_src_vec"] = batch["ptr_src_vec"].to(self.device)
            batch["src_mask"] = batch["src_mask"].to(self.device)
            batch["trg_mask"] = batch["trg_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)

        output_dict = self.model(batch)
        if self.config["generate"]:
            loss = None
        else:
            logits = output_dict["logits"]
            penalty_item = output_dict["penalty_item"]
            labels = batch["labels"].to(logits.device)
            loss = self.loss_fn(logits=logits,
                                labels=labels.to(logits.device),
                                output_mask=batch["trg_mask"].to(logits.device),
                                penalty_item=penalty_item)

        predictions = output_dict["predictions"]

        predictions = predictions.cpu().detach().numpy().tolist()
        predictions = [self.decode(prediction, src) for prediction, src in
                       zip(predictions, batch["src"])]

        metrics = self.evaluate(predictions, copy.deepcopy(batch["trg"]))

        if loss is not None:
            try:
                metrics["loss"] = loss.item()
            except:
                metrics["loss"] = loss

        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"source": batch["src"],
                                  "target": batch["trg"],
                                  "predictions": predictions},
                "loss": loss,
                "metrics": metrics,
                "stats_metrics": metrics}

        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.scheduler is not None and not self.epoch_level_scheduler:
            for group in self.optimizer.param_groups:
                group["lr"] = self.scheduler.get(self.global_step)
            self.config['current_lr'] = self.optimizer.param_groups[-1]['lr']
            self.global_step += 1
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
