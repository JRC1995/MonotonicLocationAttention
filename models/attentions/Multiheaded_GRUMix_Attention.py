import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Multiheaded_GRUMix_Attention(nn.Module):
    def __init__(self, config,
                 query_dim: int = 512, key_dim: int = 512, value_dim: int = 512, out_dim: int = 512,
                 attention_only: bool = False,
                 GRU_attention: bool = False,
                 mix_attention: bool = False,
                 location_attention_only: bool = False,
                 **kwargs):
        super(Multiheaded_GRUMix_Attention, self).__init__()

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
        self.softstair_temp = config["softstair_temp"]
        self.scaling = self.d ** (-0.5)
        self.attention_only = attention_only
        self.GRU_attention = GRU_attention
        self.mix_attention = mix_attention
        self.location_attention_only = location_attention_only
        self.eps = 1e-8

        # initialize params
        self.init_QKV()
        self.init_head_compose()
        self.init_position()

    """
    Parameter Initializers
    """

    def init_QKV(self):
        if not self.location_attention_only:
            self.query_linear = nn.Linear(self.qD, self.heads * self.d, bias=False)
            self.key_linear = nn.Linear(self.kD, self.heads * self.d, bias=False)
            T.nn.init.xavier_uniform_(self.query_linear.weight.data)
            T.nn.init.xavier_uniform_(self.key_linear.weight.data)
        if not self.attention_only:
            self.value_linear = nn.Linear(self.vD, self.heads * self.vd, bias=False)
            T.nn.init.xavier_uniform_(self.value_linear.weight.data)

    # %%
    def init_position(self):
        if self.GRU_attention:
            self.h0 = nn.Parameter(T.zeros(self.d))
            self.qcell = nn.GRUCell(input_size=self.d,
                                    hidden_size=self.d,
                                    bias=True)

        self.position_state_linear = nn.Linear(self.qD, self.heads * self.d)
        self.Wsigma = nn.Parameter(T.randn(self.heads, self.d, 1))
        if self.config["simplified"]:
            self.Wrho = nn.Parameter(T.randn(self.heads, self.d, 2))
        else:
            self.Wrho = nn.Parameter(T.randn(self.heads, self.d, 3))

        if self.mix_attention:
            self.mix_scorer = nn.Linear(self.qD, 1)
            T.nn.init.xavier_uniform_(self.mix_scorer.weight.data)

        T.nn.init.xavier_uniform_(self.position_state_linear.weight.data)
        T.nn.init.xavier_uniform_(self.Wsigma.data)
        T.nn.init.xavier_uniform_(self.Wrho.data)

    # %%
    def init_head_compose(self):
        if not self.attention_only:
            self.head_compose_linear = nn.Linear(self.heads * self.vd, self.out_dim, bias=False)
            T.nn.init.xavier_uniform_(self.head_compose_linear.weight.data)

    # %%

    def score_contents(self, Q, K):
        N, _, qS, _ = Q.size()
        vS = K.size(2)

        assert Q.size() == (N, self.heads, qS, self.d)
        assert K.size() == (N, self.heads, vS, self.d)

        Kt = K.permute(0, 1, 3, 2).contiguous()
        content_scores = T.matmul(Q, Kt)
        assert content_scores.size() == (N, self.heads, qS, vS)

        return content_scores * self.scaling

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

    def clamp(self, x,
              minimum=-float("Inf"),
              maximum=float("Inf"),
              is_leaky=True,
              negative_slope=0.01):
        """
        Clamps a tensor to the given [minimum, maximum] (leaky) bound, with
        an optional hard clamping.
        """
        lower_bound = (minimum + negative_slope * x) if is_leaky else T.zeros_like(x) + minimum
        upper_bound = (maximum + negative_slope * x) if is_leaky else T.zeros_like(x) + maximum
        clamped = T.max(lower_bound, T.min(x, upper_bound))

        return clamped

    def softstair(self, x):
        return T.floor(x) + T.sigmoid(self.softstair_temp * (T.abs(x - T.floor(x)) - 0.5))

    def score_positions(self, Q,
                        normed_value_positions,
                        past_state,
                        past_attention,
                        lengths, step_size, attention_mask):
        N, H, qS, d = Q.size()


        if self.GRU_attention:
            Q = Q.view(N * H * qS, d)
            if past_state is None:
                past_state = self.h0.view(1, d).repeat(N * H * qS, 1)
            else:
                assert past_state.size() == (N, H, qS, d)
                past_state = past_state.view(N * H * qS, d)
            state = self.qcell(F.relu(Q), past_state)
            state = state.view(N, H, qS, d)
            Q = state
        state = Q

        if past_attention is None:
            past_attention = T.zeros(N, H, qS, 1).float().to(Q.device)
            step_one = True
        else:
            step_one = False

        assert past_attention.size() == (N, H, qS, 1)

        #print("past_attention: ", past_attention)

        vS = normed_value_positions.size(-1)
        assert lengths.size() == (N, qS)

        temp = T.matmul(Q, self.Wsigma.view(1, H, d, 1))
        sigma = (F.relu(temp) + 0.27) / (lengths.view(N, 1, qS, 1))
        assert sigma.size() == (N, H, qS, 1)


        #print("steps: ", steps)
        if self.config["simplified"]:
            rho = T.matmul(Q, self.Wrho.view(1, H, d, 2))
            assert rho.size() == (N, H, qS, 2)
            steps = self.softstair(rho[..., 0]).unsqueeze(-1)
            if step_one:
                bias = T.sigmoid(rho[..., 1]).unsqueeze(-1)
            else:
                bias = 0
            mu_ = (steps * step_size) + past_attention + bias
        else:
            rho = T.matmul(Q, self.Wrho.view(1, H, d, 3))
            assert rho.size() == (N, H, qS, 3)
            steps = self.softstair(rho[..., 0].unsqueeze(-1))
            attn_gate = T.sigmoid(rho[..., 1].unsqueeze(-1))
            bias_gate = T.sigmoid(rho[..., 2].unsqueeze(-1))
            mu_ = (steps * step_size) + (attn_gate * past_attention) + (bias_gate)

        assert mu_.size() == (N, H, qS, 1)

        mu = self.clamp(mu_, minimum=0, maximum=1, is_leaky=True)
        assert mu.size() == (N, H, qS, 1)

        gauss_exp_numer = -(normed_value_positions - mu) * (normed_value_positions - mu)
        gauss_exp_denom = 2 * sigma * sigma
        assert gauss_exp_numer.size() == (N, H, qS, vS)
        assert gauss_exp_denom.size() == (N, H, qS, 1)
        gauss_exp = T.exp(gauss_exp_numer / gauss_exp_denom)

        position_attention_scores = self.sum_normalize(gauss_exp * attention_mask.float(), dim=-1)

        return position_attention_scores, state, mu

    """
    Forward Function
    """

    # %%
    def forward(self, Q, K, V,
                past_state,
                past_attention,
                attention_mask,
                value_positions,
                **kwargs):
        N, qS, _ = Q.size()
        _, vS, _ = K.size()
        assert V.size(1) == vS

        lengths = T.sum(attention_mask, dim=-1)
        assert lengths.size() == (N, qS)
        l_1 = T.max(T.ones(N, 1, qS, 1).float().to(lengths.device), lengths.view(N, 1, qS, 1) - 1)
        step_size = T.ones(N, self.heads, qS, 1).float().to(lengths.device) / l_1
        assert step_size.size() == (N, self.heads, qS, 1)

        normed_value_positions = value_positions.view(1, 1, 1, vS).repeat(N, self.heads, 1, 1) * step_size
        assert normed_value_positions.size() == (N, self.heads, qS, vS)

        if self.mix_attention:
            mix_score = T.sigmoid(self.mix_scorer(Q))
            assert mix_score.size() == (N, qS, 1)
            mix_score = mix_score.unsqueeze(1)
            assert mix_score.size() == (N, 1, qS, 1)

        Qpos = self.position_state_linear(Q)
        Qpos = Qpos.view(N, qS, self.heads, self.d)
        Qpos = Qpos.permute(0, 2, 1, 3).contiguous()
        assert Qpos.size() == (N, self.heads, qS, self.d)

        if not self.location_attention_only:
            Q = self.query_linear(Q)
            K = self.key_linear(K)
            assert Q.size() == (N, qS, self.heads * self.d)
            Q = Q.view(N, qS, self.heads, self.d)
            K = K.view(N, vS, self.heads, self.d)
            Q = Q.permute(0, 2, 1, 3).contiguous()
            K = K.permute(0, 2, 1, 3).contiguous()

        if not self.attention_only:
            V = self.value_linear(V)
            assert V.size() == (N, vS, self.heads * self.vd)
            V = V.view(N, vS, self.heads, self.vd)
            V = V.permute(0, 2, 1, 3).contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            assert attention_mask.size() == (N, 1, qS, vS)
            attention_mask = attention_mask.bool()

        if not self.location_attention_only:
            content_scores = self.score_contents(Q, K)
            content_attention_scores = self.masked_softmax(content_scores, mask=attention_mask, dim=-1)

        position_attention_scores, state, mu = self.score_positions(Q=Qpos,
                                                                normed_value_positions=normed_value_positions,
                                                                past_attention=past_attention,
                                                                past_state=past_state,
                                                                lengths=lengths,
                                                                step_size=step_size,
                                                                attention_mask=attention_mask)

        if self.location_attention_only:
            attention_scores = position_attention_scores
        elif self.mix_attention:
            attention_scores = mix_score * position_attention_scores + (1-mix_score) * content_attention_scores

        past_state = state
        attention_dist = attention_scores.mean(1)

        if not self.attention_only:
            attention_scores = F.dropout(attention_scores, p=self.attn_dropout, training=self.training)

            attended_values = T.matmul(attention_scores, V)
            if self.config["position_past"] or self.location_attention_only:
                past_attention = T.sum(position_attention_scores * normed_value_positions, dim=-1).unsqueeze(-1)
            else:
                past_attention = T.sum(attention_scores * normed_value_positions, dim=-1).unsqueeze(-1)
            assert past_attention.size() == (N, self.heads, qS, 1)

            assert attended_values.size() == (N, self.heads, qS, self.vd)

            attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
            attended_values = attended_values.view(N, qS, self.heads * self.vd)

            attended_values = self.head_compose_linear(attended_values)
            assert attended_values.size() == (N, qS, self.out_dim)
        else:
            attended_values = None
            past_attention = None

        return {"attended_values": attended_values,
                "attention_dist": attention_dist,
                "full_attention_dist": attention_scores,
                "past_state": past_state,
                "past_attention": past_attention}
