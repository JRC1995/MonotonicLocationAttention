import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.utils import generate_relative_positional_embeddings, RoFormerSinusoidalPositionalEmbedding

class Multiheaded_Attention(nn.Module):
    def __init__(self, config,
                 query_dim: int = 512, key_dim: int = 512, value_dim: int = 512, out_dim: int = 512,
                 relative_pos: bool = True,
                 attention_only: bool = False,
                 forward_positions: bool = True,
                 reverse_positions: bool = False,
                 mix_forward_reverse: bool = False,
                 **kwargs):
        super(Multiheaded_Attention, self).__init__()

        self.qD = query_dim
        self.kD = key_dim
        self.vD = value_dim
        self.out_dim = out_dim

        self.config = config
        self.heads = config["heads"]
        self.d = config["qk_head_dim"]
        self.vd = config["v_head_dim"]
        self.attn_dropout = config["attn_dropout"]
        self.position_max_len = config["position_max_len"]
        self.rope = config["rope"]
        self.scaling = self.d ** (-0.5)
        self.relative_pos = relative_pos
        self.attention_only = attention_only
        self.eps = 1e-8
        if self.relative_pos:
            if self.rope:
                self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
                    config["position_max_len"], self.d
                )
            else:
                self.relative_position_embed = generate_relative_positional_embeddings(max_len=config["position_max_len"],
                                                                                       d_model=self.qD)

        # initialize params
        self.init_QKV()
        self.init_head_compose()
        self.init_position()

    """
    Parameter Initializers
    """

    def init_QKV(self):
        self.query_linear = nn.Linear(self.qD, self.heads * self.d, bias=False)
        self.key_linear = nn.Linear(self.kD, self.heads * self.d, bias=False)
        T.nn.init.xavier_uniform_(self.query_linear.weight.data)
        T.nn.init.xavier_uniform_(self.key_linear.weight.data)
        if not self.attention_only:
            self.value_linear = nn.Linear(self.vD, self.heads * self.vd, bias=False)
            T.nn.init.xavier_uniform_(self.value_linear.weight.data)

    # %%
    def init_position(self):
        if self.relative_pos and not self.rope:
            self.content_bias = nn.Parameter(T.zeros(self.heads, self.d))
            self.forward_position_bias = nn.Parameter(T.zeros(self.heads, self.d))
            self.forward_position_linear = nn.Linear(self.qD, self.heads * self.d, bias=False)
            T.nn.init.xavier_uniform_(self.forward_position_linear.weight.data)

    # %%
    def init_head_compose(self):
        if not self.attention_only:
            self.head_compose_linear = nn.Linear(self.heads * self.vd, self.out_dim, bias=False)
            T.nn.init.xavier_uniform_(self.head_compose_linear.weight.data)

    # Taken from Huggingface
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, state):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = T.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = T.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_state = T.stack([-state[..., 1::2], state[..., ::2]], dim=-1).reshape_as(state)
        state = state * cos_pos + rotate_half_state * sin_pos
        return state



    # %%

    def score_contents(self, Q, K):
        N, _, qS, _ = Q.size()
        vS = K.size(2)

        assert Q.size() == (N, self.heads, qS, self.d)
        assert K.size() == (N, self.heads, vS, self.d)

        if self.relative_pos and not self.rope:
            u = self.content_bias.view(1, self.heads, 1, self.d)
        else:
            u = 0

        Kt = K.permute(0, 1, 3, 2).contiguous()
        content_scores = T.matmul(Q + u, Kt)
        assert content_scores.size() == (N, self.heads, qS, vS)

        return content_scores * self.scaling

    # %%
    def score_forward_positions(self, Q, query_positions, value_positions):
        N, H, qS, d = Q.size()
        assert query_positions.size(0) == qS
        vS = value_positions.size(0)
        """
        position_idx = T.arange(S).unsqueeze(0).repeat(S, 1)
        position_idx_t = position_idx.permute(1, 0).contiguous()
        relative_mat_idx = position_idx - position_idx_t + self.position_max_len
        if query_pos is None:
            relative_mat_idx = relative_mat_idx[0:qS, 0:vS]
        else:
            relative_mat_idx = relative_mat_idx[query_pos, 0:vS].unsqueeze(0)
        """

        relative_mat_idx = value_positions.unsqueeze(-2) - query_positions.unsqueeze(-1) + self.position_max_len
        #print("forward: ", relative_mat_idx - self.position_max_len)
        relative_mat_idx = relative_mat_idx.long()
        assert relative_mat_idx.size() == (qS, vS)

        RE = self.relative_position_embed(relative_mat_idx)
        assert RE.size() == (qS, vS, self.qD)
        RE = self.forward_position_linear(RE)
        assert RE.size() == (qS, vS, self.heads * self.d)

        RE = RE.view(qS, vS, self.heads, self.d)
        RE = RE.permute(2, 0, 1, 3).contiguous()
        assert RE.size() == (self.heads, qS, vS, self.d)

        REt = RE.permute(0, 1, 3, 2).contiguous()
        assert REt.size() == (self.heads, qS, self.d, vS)

        assert Q.size() == (N, H, qS, d)
        Q = Q.permute(1, 2, 0, 3).contiguous()
        assert Q.size() == (H, qS, N, d)

        v = self.forward_position_bias.view(self.heads, 1, 1, d)
        position_scores = T.matmul(Q + v, REt)

        assert position_scores.size() == (H, qS, N, vS)
        position_scores = position_scores.permute(2, 0, 1, 3).contiguous()
        assert position_scores.size() == (N, H, qS, vS)

        return position_scores * self.scaling


    """
    Forward Function
    """

    def masked_softmax(self, logits, mask, dim):
        if mask is None:
            return F.softmax(logits, dim=dim)

        logits = logits.masked_fill(~mask, float("-inf"))
        logits = F.softmax(logits, dim=dim)
        return logits

    # %%
    def sum_normalize(self, logits, dim=-1):
        eps = 1e-20
        return logits / T.sum(logits + eps, keepdim=True, dim=dim)

    # %%
    def forward(self, Q, K, V,
                attention_mask,
                query_positions, value_positions,
                **kwargs):
        N, qS, _ = Q.size()
        _, vS, _ = K.size()
        assert V.size(1) == vS

        if self.rope and self.relative_pos:
            query_sinusoidal_pos = self.embed_positions(query_positions)[None, None, :, :]
            assert query_sinusoidal_pos.size() == (1, 1, qS, self.d)
            key_sinusoidal_pos = self.embed_positions(value_positions)[None, None, :, :]
            assert key_sinusoidal_pos.size() == (1, 1, vS, self.d)



        Q = self.query_linear(Q)
        K = self.key_linear(K)
        if not self.attention_only:
            V = self.value_linear(V)

        assert Q.size() == (N, qS, self.heads * self.d)

        if not self.attention_only:
            assert V.size() == (N, vS, self.heads * self.vd)
            V = V.view(N, vS, self.heads, self.vd)
            V = V.permute(0, 2, 1, 3).contiguous()

        Q = Q.view(N, qS, self.heads, self.d)
        K = K.view(N, vS, self.heads, self.d)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            assert attention_mask.size() == (N, 1, qS, vS)
            attention_mask = attention_mask.bool()

        if self.relative_pos and self.rope:
            Q = self.apply_rotary_position_embeddings(query_sinusoidal_pos, Q)
            K = self.apply_rotary_position_embeddings(key_sinusoidal_pos, K)

        content_scores = self.score_contents(Q, K)

        if self.relative_pos and not self.rope:
            """
            print("forward: ", self.forward_positions)
            print("backward: ", self.reverse_positions)
            print("mix: ", self.mix_forward_reverse)
            """

            forward_position_scores = self.score_forward_positions(Q, query_positions, value_positions)
            forward_edge_scores = content_scores + forward_position_scores
            forward_attention_scores = self.masked_softmax(forward_edge_scores, mask=attention_mask, dim=-1)

            attention_scores = forward_attention_scores.clone()
        else:
            edge_scores = content_scores
            attention_scores = self.masked_softmax(edge_scores, mask=attention_mask, dim=-1)

        attention_dist = attention_scores.mean(1)

        if not self.attention_only:
            attention_scores = F.dropout(attention_scores, p=self.attn_dropout, training=self.training)

            attended_values = T.matmul(attention_scores, V)

            assert attended_values.size() == (N, self.heads, qS, self.vd)

            attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
            attended_values = attended_values.view(N, qS, self.heads * self.vd)

            attended_values = self.head_compose_linear(attended_values)
            assert attended_values.size() == (N, qS, self.out_dim)
        else:
            attended_values = None

        return {"attended_values": attended_values,
                "attention_dist": attention_dist,
                "full_attention_dist": attention_scores}
