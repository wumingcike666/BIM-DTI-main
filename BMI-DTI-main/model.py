import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * self.softmax(x) + x


class AttentionDTA(nn.Module):
    def __init__(self, protein_MAX_LENGH=1200, protein_kernel=[4, 8, 12],
                 drug_MAX_LENGH=100, drug_kernel=[4, 6, 8],
                 conv=64, char_dim=128, head_num=4, dropout_rate=0.1,
                 tf_layers=2, tf_heads=4, tf_ffn=256, tf_dropout=0.1):
        super(AttentionDTA, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel


        self.tf_layers = tf_layers
        self.tf_heads = tf_heads
        self.tf_ffn = tf_ffn
        self.tf_dropout = tf_dropout

        self.feature_dim = self.conv * 3  

        assert self.feature_dim % self.head_num == 0, \
            f"feature_dim={self.feature_dim} 必须能被 head_num={self.head_num} 整除（用于显式双向MHA）"

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 3, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(self.conv * 3, self.conv * 3),
            nn.ReLU()
        )

        self.self_attention = nn.MultiheadAttention(self.conv * 3, 4, batch_first=True)

        self.drug_to_protein_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.head_num,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.protein_to_drug_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.head_num,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.drug_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.Sigmoid()
        )
        self.protein_gate = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.Sigmoid()
        )

        self.drug_fuse_norm = nn.LayerNorm(self.feature_dim)
        self.protein_fuse_norm = nn.LayerNorm(self.feature_dim)

        self.drug_fuse_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.protein_fuse_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.conv * 6, 1024),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def _conv_out_len(self, L, kernels):
        for k in kernels:
            L = L - k + 1
        return L

    def _build_cnn_padding_mask(self, raw_mask, kernels):
        """
        raw_mask: [B, L_raw], 1=有效token, 0=padding
        return:   [B, L_cnn], True=padding, False=有效
        """
        if raw_mask.dim() == 3:
            raw_mask = raw_mask.squeeze(-1)

        raw_mask = raw_mask.float()
        B, L_raw = raw_mask.shape
        L_cnn = self._conv_out_len(L_raw, kernels)

        valid_lens = raw_mask.sum(dim=1).long()
        valid_lens_cnn = valid_lens.clone()

        for k in kernels:
            valid_lens_cnn = valid_lens_cnn - k + 1

        valid_lens_cnn = torch.clamp(valid_lens_cnn, min=1, max=L_cnn)

        idx = torch.arange(L_cnn, device=raw_mask.device).unsqueeze(0).expand(B, -1)
        key_padding_mask = idx >= valid_lens_cnn.unsqueeze(1)

        return key_padding_mask

    def fu_attention(self, atoms_vector, amino_vector, compound_mask, protein_mask,
                     drug_key_padding_mask=None, protein_key_padding_mask=None):
        """
        atoms_vector: [B, C, Ld]
        amino_vector: [B, C, Lp]
        return: iner [B, L_fused, C]
        """

        amino_vector = amino_vector.permute(0, 2, 1).contiguous()   # [B, Lp, C]
        atoms_vector = atoms_vector.permute(0, 2, 1).contiguous()   # [B, Ld, C]

        drug_cross, attn_d2t = self.drug_to_protein_attn(
            query=atoms_vector,
            key=amino_vector,
            value=amino_vector,
            key_padding_mask=protein_key_padding_mask
        )   # [B, Ld, C]
        drug_cross = torch.nan_to_num(drug_cross, nan=0.0, posinf=1e4, neginf=-1e4)

        protein_cross, attn_t2d = self.protein_to_drug_attn(
            query=amino_vector,
            key=atoms_vector,
            value=atoms_vector,
            key_padding_mask=drug_key_padding_mask
        )   # [B, Lp, C]
        protein_cross = torch.nan_to_num(protein_cross, nan=0.0, posinf=1e4, neginf=-1e4)

        self.last_attn_d2t = attn_d2t.detach().cpu() if attn_d2t is not None else None
        self.last_attn_t2d = attn_t2d.detach().cpu() if attn_t2d is not None else None
        self.last_drug_key_padding_mask = drug_key_padding_mask.detach().cpu()
        self.last_protein_key_padding_mask = protein_key_padding_mask.detach().cpu()

        drug_gate = self.drug_gate(torch.cat([atoms_vector, drug_cross], dim=-1))
        drug_gate = torch.nan_to_num(drug_gate, nan=0.5, posinf=1.0, neginf=0.0)
        self.last_drug_gate = drug_gate.detach().float().cpu()
        drug_fused = drug_gate * drug_cross + (1.0 - drug_gate) * atoms_vector
        drug_fused = self.drug_fuse_norm(drug_fused + self.drug_fuse_ffn(drug_fused))
        drug_fused = torch.nan_to_num(drug_fused, nan=0.0, posinf=1e4, neginf=-1e4)

        protein_gate = self.protein_gate(torch.cat([amino_vector, protein_cross], dim=-1))
        protein_gate = torch.nan_to_num(protein_gate, nan=0.5, posinf=1.0, neginf=0.0)
        self.last_target_gate = protein_gate.detach().float().cpu()
        protein_fused = protein_gate * protein_cross + (1.0 - protein_gate) * amino_vector
        protein_fused = self.protein_fuse_norm(protein_fused + self.protein_fuse_ffn(protein_fused))
        protein_fused = torch.nan_to_num(protein_fused, nan=0.0, posinf=1e4, neginf=-1e4)


        branch_tokens = torch.cat([protein_fused, drug_fused], dim=1)   # [B, Lp+Ld, C]
        branch_tokens = torch.nan_to_num(branch_tokens, nan=0.0, posinf=1e4, neginf=-1e4)

 
        iner = self.fc(branch_tokens)   # [B, Lp+Ld, C]
        iner = torch.nan_to_num(iner, nan=0.0, posinf=1e4, neginf=-1e4)


        fused_padding_mask = torch.cat([protein_key_padding_mask, drug_key_padding_mask], dim=1)
        iner_f, _ = self.self_attention(iner, iner, iner, key_padding_mask=fused_padding_mask)
        iner_f = torch.nan_to_num(iner_f, nan=0.0, posinf=1e4, neginf=-1e4)

        iner = iner + iner_f
        iner = iner.masked_fill(fused_padding_mask.unsqueeze(-1), 0.0)
        iner = torch.nan_to_num(iner, nan=0.0, posinf=1e4, neginf=-1e4)

        return iner

    def forward(self, drug, protein, atom, node, edge_index, edge_attr, compound_mask, protein_mask):
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)

        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)           # [B, C, Ld]
        proteinConv = self.Protein_CNNs(proteinembed) # [B, C, Lp]

        drug_raw_mask = (drug != 0).float()
        protein_raw_mask = (protein != 0).float()

        drug_key_padding_mask = self._build_cnn_padding_mask(drug_raw_mask, self.drug_kernel)
        protein_key_padding_mask = self._build_cnn_padding_mask(protein_raw_mask, self.protein_kernel)

        pair = self.fu_attention(
            drugConv, proteinConv, drug_raw_mask, protein_raw_mask,
            drug_key_padding_mask=drug_key_padding_mask,
            protein_key_padding_mask=protein_key_padding_mask
        )

        fused_valid_mask = torch.cat(
            [~protein_key_padding_mask, ~drug_key_padding_mask], dim=1
        ).float()   # [B, L], 1=valid, 0=padding


        pair_for_max = pair.masked_fill(fused_valid_mask.unsqueeze(-1) == 0, -1e4)
        d_max_feature_a = pair_for_max.max(dim=1).values   # [B, 3C]

        pair_for_avg = pair * fused_valid_mask.unsqueeze(-1)
        valid_count = fused_valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        d_avg_feature_a = pair_for_avg.sum(dim=1) / valid_count

        drug_feature_a = torch.cat([d_max_feature_a, d_avg_feature_a], dim=1)  # [B, 6C]

        drug_feature_a = torch.nan_to_num(drug_feature_a, nan=0.0, posinf=0.0, neginf=0.0)
        predict = self.classifier(drug_feature_a)
        predict = torch.nan_to_num(predict, nan=0.0, posinf=0.0, neginf=0.0)
        return predict
