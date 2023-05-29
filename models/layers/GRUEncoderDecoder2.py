import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.attentions import *


class TiedEmbedding(T.nn.Module):
    def __init__(self, weights: T.Tensor):
        super().__init__()

        # Hack: won't save it as a parameter
        self.w = [weights]
        self.bias = T.nn.Parameter(T.zeros(self.w[0].shape[0]))

    def forward(self, t: T.Tensor) -> T.Tensor:
        return F.linear(t, self.w[0], self.bias)


class GRUEncoderDecoder2(nn.Module):
    def __init__(self, config):
        super(GRUEncoderDecoder2, self).__init__()

        self.pad_inf = -1.0e10
        self.vocab_len = config["vocab_len"]
        self.config = config
        self.dropout = config["dropout"]
        self.UNK_id = config["UNK_id"]
        self.SOS_id = config["SOS_id"]
        self.EOS_id = config["EOS_id"]
        self.embed_dim = config["embd_dim"]
        self.inter_value_size = config["inter_value_size"]
        self.attention_type = config["attention_type"]

        self.embed_layer = nn.Embedding(self.vocab_len, config["embd_dim"],
                                        padding_idx=config["PAD_id"])

        self.encoder_hidden_size = config["encoder_hidden_size"]

        self.encoder = nn.GRU(input_size=config["embd_dim"],
                              hidden_size=config["encoder_hidden_size"],
                              num_layers=config["encoder_layers"],
                              batch_first=True,
                              bidirectional=self.config["bidirectional"])

        self.k = 2 if self.config["bidirectional"] else 1
        self.decoder_hidden_size = self.k * self.encoder_hidden_size
        self.START = nn.Parameter(T.zeros(config["embd_dim"]))
        self.END = nn.Parameter(T.zeros(config["embd_dim"]))
        self.h0 = nn.Parameter(T.zeros(self.decoder_hidden_size))
        self.value_transform = nn.Linear(self.decoder_hidden_size, self.inter_value_size)

        self.decodercell1 = nn.GRUCell(input_size=config["embd_dim"] + config["decoder_hidden_size"],
                                       hidden_size=config["decoder_hidden_size"],
                                       bias=True)
        self.decodercell2 = nn.GRUCell(input_size=config["decoder_hidden_size"],
                                       hidden_size=config["decoder_hidden_size"],
                                       bias=True)

        if self.config["mix_forward_reverse"]:
            self.dir_gater = nn.Linear(self.decoder_hidden_size, 1)

        self.attention_layer = eval(self.attention_type)(query_dim=self.decoder_hidden_size,
                                                         key_dim=self.decoder_hidden_size,
                                                         value_dim=self.inter_value_size,
                                                         out_dim=self.decoder_hidden_size,
                                                         config=config,
                                                         relative_pos=config["cross_relative_pos"],
                                                         forward_positions=config["forward_positions"],
                                                         reverse_positions=config["reverse_positions"],
                                                         mix_forward_reverse=config["mix_forward_reverse"],
                                                         GRU_attention=config["GRU_attention"],
                                                         mix_attention=config["mix_attention"],
                                                         location_attention_only=config["location_attention_only"])
        self.out_linear1 = nn.Linear(self.decoder_hidden_size, self.embed_dim)
        self.output_map = TiedEmbedding(
            self.embed_layer.weight)  # nn.Linear(self.decoder_hidden_size, self.vocab_len)  # TiedEmbedding(self.embed_layer.weight)
        self.eps = 1e-9

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.embed_layer.weight.data, mode="fan_out")
        nn.init.xavier_uniform_(self.value_transform.weight.data)
        nn.init.xavier_uniform_(self.out_linear1.weight.data)
        # nn.init.kaiming_normal_(self.CLS.data, mode="fan_out")

    # %%
    def augment_sequence(self, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S)
        input_mask = input_mask.unsqueeze(-1)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        # ADD START TOKEN
        START = self.START.view(1, 1, D).repeat(N, 1, 1)
        sequence = T.cat([START, sequence], dim=1)
        assert sequence.size() == (N, S + 1, D)
        input_mask = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask], dim=1)
        assert input_mask.size() == (N, S + 1, 1)

        # ADD END TOKEN
        input_mask_no_end = T.cat([input_mask.clone(), T.zeros(N, 1, 1).float().to(input_mask.device)], dim=1)
        input_mask_yes_end = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask.clone()], dim=1)
        END_mask = input_mask_yes_end - input_mask_no_end
        assert END_mask.size() == (N, S + 2, 1)

        END = self.END.view(1, 1, D).repeat(N, S + 2, 1)
        sequence = T.cat([sequence, T.zeros(N, 1, D).float().to(sequence.device)], dim=1)
        sequence = END_mask * END + (1 - END_mask) * sequence

        input_mask = input_mask_yes_end
        input_mask_no_start = T.cat([T.zeros(N, 1, 1).float().to(input_mask.device),
                                     input_mask[:, 1:, :]], dim=1)

        return sequence, input_mask.squeeze(-1), END_mask.squeeze(-1), input_mask_no_start.squeeze(
            -1), input_mask_no_end.squeeze(-1)

    # %%
    def forward(self, src_idx, ptr_src_idx, max_oov_num, input_mask, trg_idx=None, output_mask=None):

        src = self.embed_layer(src_idx)
        src, input_mask, _, _, _ = self.augment_sequence(src, input_mask)

        N, S1, D = src.size()
        assert input_mask.size() == (N, S1)

        if trg_idx is not None:
            N, S2 = trg_idx.size()
        else:
            S2 = self.config["max_decode_len"]

        """
        ENCODING
        """
        lengths = T.sum(input_mask, dim=1).long().view(N).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)
        encoded_src, hn = self.encoder(packed_sequence)
        encoded_src, _ = nn.utils.rnn.pad_packed_sequence(encoded_src, batch_first=True)
        assert encoded_src.size() == (N, S1, self.k * self.encoder_hidden_size)

        assert hn.size() == (self.k * 2, N, self.encoder_hidden_size)
        hn = hn.view(self.k, 2, N, self.encoder_hidden_size)[:, 1, ...]
        hn = hn.permute(1, 0, 2).contiguous()
        assert hn.size() == (N, self.k, self.encoder_hidden_size)
        hn = hn.view(N, self.k * self.encoder_hidden_size)

        encoded_src = F.dropout(encoded_src, p=self.config["dropout"], training=self.training)

        """
        PREPARING FOR DECODING
        """
        h1 = hn
        h = hn  # self.h0.view(1, self.decoder_hidden_size).repeat(N, 1)
        input_id = T.ones(N).long().to(src.device) * self.config["vocab2idx"]["<sos>"]
        output_dists = []
        value_positions = T.arange(S1).to(h.device)
        past_attention = None
        past_state = None

        if self.config["reverse_positions"]:
            reverse_encoded_src = T.flip(encoded_src, dims=[1])
            count_zeros = T.sum(1 - input_mask, dim=-1).view(N)
            reverse_encoded_src = T.cat([reverse_encoded_src,
                                         T.zeros(N, S1, self.k * self.encoder_hidden_size).float().to(h.device)], dim=1)
            new_batch_stack = []
            for i in range(N):
                start_id = count_zeros[i].long().item()
                new_batch_stack.append(reverse_encoded_src[i, start_id:start_id + S1, :])
            reverse_encoded_src = T.stack(new_batch_stack, dim=0)

        if self.config["mix_forward_reverse"]:
            # encoded_src = reverse_encoded_src
            # input_mask = reverse_input_mask
            g = T.sigmoid(5 * self.dir_gater(h)).view(N, 1, 1)
            encoded_src = g * encoded_src + (1 - g) * reverse_encoded_src

        elif self.config["reverse_positions"]:
            encoded_src = reverse_encoded_src

        key_encoded_src = encoded_src.clone()
        value_encoded_src = F.leaky_relu(self.value_transform(encoded_src))

        for t in range(S2):
            # print("time: ", t)
            query_positions = T.tensor([t]).to(h.device)
            if t > 0 and not self.config["generate"]:
                input_id = trg_idx[:, t - 1]
            input = self.embed_layer(input_id)
            assert input.size() == (N, self.embed_dim)

            Q = h.unsqueeze(1)
            assert Q.size() == (N, 1, self.decoder_hidden_size)

            d = self.attention_layer(K=key_encoded_src,
                                     V=value_encoded_src,
                                     Q=Q,
                                     query_positions=query_positions,
                                     value_positions=value_positions,
                                     attention_mask=input_mask.unsqueeze(1),
                                     past_state=past_state,
                                     past_attention=past_attention)
            c = d["attended_values"]
            if "past_attention" in d:
                past_attention = d["past_attention"]
            if "past_state" in d:
                past_state = d["past_state"]
            assert c.size() == (N, 1, self.decoder_hidden_size)
            c = c.squeeze(1)

            decoder_input = T.cat([input, c], dim=-1)
            assert decoder_input.size() == (N, self.embed_dim + self.decoder_hidden_size)

            h1 = self.decodercell1(decoder_input, h1)
            h = self.decodercell2(h1, h)
            out_state = self.out_linear1(h)

            gen_dist = F.softmax(self.output_map(out_state), dim=-1)
            assert gen_dist.size() == (N, self.config["vocab_len"])
            output_dists.append(gen_dist)
            input_id = T.argmax(gen_dist, dim=-1)

        output_dists = T.stack(output_dists, dim=1)
        assert output_dists.size() == (N, S2, self.config["vocab_len"])

        return {"logits": output_dists,
                "penalty_item": None,
                "predictions": T.argmax(output_dists, dim=-1)}
