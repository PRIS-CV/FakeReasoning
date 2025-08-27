from .net_utils import MLP, LN

import torch.nn as nn
import torch.nn.functional as F
import torch, math

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, config):
        super(MHAtt, self).__init__()
        self.config = config

        self.linear_v = nn.Linear(config["HIDDEN_SIZE"], config["HIDDEN_SIZE"])
        self.linear_k = nn.Linear(config["HIDDEN_SIZE"], config["HIDDEN_SIZE"])
        self.linear_q = nn.Linear(config["HIDDEN_SIZE"], config["HIDDEN_SIZE"])
        self.linear_merge = nn.Linear(config["HIDDEN_SIZE"], config["HIDDEN_SIZE"])

        self.dropout = nn.Dropout(config["DROPOUT_R"])

    def forward(self, v, k, q, attn_bias=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.config["MULTI_HEAD"],
            self.config["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.config["MULTI_HEAD"],
            self.config["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.config["MULTI_HEAD"],
            self.config["HIDDEN_SIZE_HEAD"]
        ).transpose(1, 2)

        atted = self.att(v, k, q, attn_bias)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.config["HIDDEN_SIZE"]
        )
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, attn_bias=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if attn_bias is not None:
            scores = scores + attn_bias  
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)



# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, config):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=config["HIDDEN_SIZE"],
            mid_size=config["FF_SIZE"],
            out_size=config["HIDDEN_SIZE"],
            dropout_r=config["DROPOUT_R"],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
    

class AttentionBiasMLP(nn.Module):
    def __init__(self, channels=16, hidden_channels=32):
        super(AttentionBiasMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),  # 16 → 32
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1)   # 32 → 16
        )

    def forward(self, x):  # x: [n, 16, 576, 576]
        return self.mlp(x)  # output: [n, 16, 576, 576]


class CrossAttention(nn.Module):
    def __init__(self, config, delay_load=True):
        super(CrossAttention, self).__init__()
        self.is_loaded = False
        self.num_layers = 1

        if not delay_load:
            self.config = config
            self.load_attention_model()
        else:
            self.config = config

    def load_attention_model(self):
        self.attention_mlp = AttentionBiasMLP()
        self.mhatt = nn.ModuleList([MHAtt(self.config) for _ in range(self.num_layers)])
        self.ffn = nn.ModuleList([FFN(self.config) for _ in range(self.num_layers)])
        self.norm = nn.ModuleList([LN(self.config["HIDDEN_SIZE"]) for _ in range(self.num_layers * 2)])
        self.dropout = nn.ModuleList([nn.Dropout(self.config["DROPOUT_R"]) for _ in range(self.num_layers * 2)])

        self.requires_grad_(True)
        self.is_loaded = True


    def build_attention_bias(self, attn_map): #[n, 16, 577, 577]
        attn_map = attn_map[:, :, 1:, 1:] #[n, 16, 576, 576]
        attn_bias = self.attention_mlp(torch.log(attn_map + 1e-6))

        return attn_bias
    

    def forward(self, q, kv, attn_map=None):
        if not self.is_loaded:
            raise RuntimeError("CrossAttention module not loaded. Please call load_attention_model() first.")

        if attn_map is not None:
            attn_bias = self.build_attention_bias(attn_map)
        else:
            attn_bias = None

        for i in range(self.num_layers):
            q = self.norm[2*i](q + self.dropout[2*i](self.mhatt[i](kv, kv, q, attn_bias)))
            q = self.norm[2*i + 1](q + self.dropout[2*i + 1](self.ffn[i](q)))

        return q