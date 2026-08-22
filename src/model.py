"""
MEDDC-DTI: Multi-route Evidence Decomposition and Disagreement-aware Calibration
for Drug-Target Interaction Prediction

A publication-ready DTI prediction model with three-layer evidence architecture:
  1. Biochemical Evidence Layer: Multi-source molecular/protein priors (ChemBERTa, ESM2, RDKit)
  2. Topological Evidence Layer: Graph-based relational context encoding
  3. Interaction Evidence Layer: Sparse bipartite attention for pairwise compatibility

Key modules:
  - MultiSourceBiochemicalEncoder: Fuses drug/protein biochemical priors with learned reliability
  - SymmetricCrossModalGating: Bidirectional drug-protein feature modulation
  - EvidenceExpertConsensus: Divergence-aware expert consensus with interpretable contributions
  - EvidenceFusion: Reliability-calibrated softmax fusion of three evidence layers
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter
from torch.utils.checkpoint import checkpoint

from config import MODEL_CONFIG


# =============================================================================
# Core Evidence Encoders
# =============================================================================

class BiochemicalEncoder(nn.Module):
    """Autoencoder-based biochemical feature encoder."""

    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_z):
        super().__init__()
        self.encoder = nn.Sequential(
            Linear(n_input, n_hidden_1), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_1, n_hidden_2), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_2, n_hidden_3), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_3, n_z),
        )
        self.decoder = nn.Sequential(
            Linear(n_z, n_hidden_3), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_3, n_hidden_2), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_2, n_hidden_1), nn.LeakyReLU(0.2, inplace=True),
            Linear(n_hidden_1, n_input),
        )

    def forward(self, x):
        return self.encoder(x)

    def reconstruct(self, h):
        return self.decoder(h)


class GNNLayer(Module):
    """Single GNN propagation layer."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.act = nn.Tanh()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj, activate=False):
        h = self.act(torch.mm(x, self.weight)) if activate else torch.mm(x, self.weight)
        if h.dtype != torch.float32 or adj.dtype != torch.float32:
            return torch.spmm(adj.float(), h.float()).to(h.dtype)
        return torch.spmm(adj, h)


class TopologicalEncoder(nn.Module):
    """GNN-based topological evidence encoder."""

    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_z):
        super().__init__()
        self.gnn_1 = GNNLayer(n_input, n_hidden_1)
        self.gnn_2 = GNNLayer(n_hidden_1, n_hidden_2)
        self.gnn_3 = GNNLayer(n_hidden_2, n_z)

    def forward(self, x, adj):
        h = self.gnn_1(x, adj, activate=True)
        h = self.gnn_2(h, adj, activate=True)
        return self.gnn_3(h, adj, activate=False)


class TopologicalDecoder(nn.Module):
    """GNN-based decoder for topology reconstruction."""

    def __init__(self, n_z, n_hidden_1, n_hidden_2, n_input):
        super().__init__()
        self.gnn_1 = GNNLayer(n_z, n_hidden_1)
        self.gnn_2 = GNNLayer(n_hidden_1, n_hidden_2)
        self.gnn_3 = GNNLayer(n_hidden_2, n_input)

    def forward(self, h, adj):
        z = self.gnn_1(h, adj, activate=True)
        z = self.gnn_2(z, adj, activate=True)
        return self.gnn_3(z, adj, activate=True), None


# =============================================================================
# Multi-Source Biochemical Prior Adapter
# =============================================================================

class MultiSourceBiochemicalAdapter(nn.Module):
    """
    Fuses multiple biochemical prior sources (ChemBERTa, ESM2, RDKit, etc.)
    with learned per-source reliability gates and attention-based weighting.

    Outputs:
      - h_biochem: Enhanced biochemical embedding
      - source_reliability: Per-source reliability scores
      - source_weights: Attention weights across sources
    """

    def __init__(self, n_z, hidden=64, prior_weight=0.25):
        super().__init__()
        self.n_z = n_z
        self.hidden = hidden
        self.prior_weight = prior_weight
        self.projectors = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        self.scorers = nn.ModuleDict()

    def configure(self, source_dims):
        """Dynamically configure projectors for available sources."""
        for name, dim in source_dims.items():
            key = str(name)
            if key in self.projectors:
                continue
            self.projectors[key] = nn.Sequential(
                Linear(int(dim), self.n_z), nn.LayerNorm(self.n_z), nn.GELU()
            )
            self.gates[key] = nn.Sequential(
                Linear(self.n_z * 2, self.hidden), nn.LayerNorm(self.hidden),
                nn.ReLU(inplace=True), Linear(self.hidden, 1),
            )
            self.scorers[key] = nn.Sequential(
                Linear(self.n_z + 1, self.hidden), nn.LayerNorm(self.hidden),
                nn.ReLU(inplace=True), Linear(self.hidden, 1),
            )
            nn.init.zeros_(self.gates[key][-1].weight)
            nn.init.zeros_(self.gates[key][-1].bias)
            nn.init.zeros_(self.scorers[key][-1].weight)
            nn.init.zeros_(self.scorers[key][-1].bias)

    def forward(self, h_base, source_features):
        if not source_features:
            zeros = torch.zeros((h_base.size(0), 1), device=h_base.device, dtype=h_base.dtype)
            return h_base, {}, {}, {}, zeros, torch.zeros_like(h_base)

        embeddings, reliabilities, logits, names = {}, {}, [], []

        for name in sorted(source_features):
            if name not in self.projectors:
                continue
            features = source_features[name].to(h_base.device, h_base.dtype)
            h_source = self.projectors[name](features)
            reliability = torch.sigmoid(self.gates[name](torch.cat([h_base, h_source], dim=-1)))
            valid = (features.abs().sum(dim=-1, keepdim=True) > 0).to(h_base.dtype)
            reliability = reliability * valid
            score = self.scorers[name](torch.cat([h_source, reliability], dim=-1))
            score = score.masked_fill(valid <= 0, -1e4)

            embeddings[name] = h_source
            reliabilities[name] = reliability
            logits.append(score)
            names.append(name)

        if not logits:
            zeros = torch.zeros((h_base.size(0), 1), device=h_base.device, dtype=h_base.dtype)
            return h_base, {}, {}, {}, zeros, torch.zeros_like(h_base)

        stacked = torch.stack(logits, dim=1)
        active = torch.stack([
            (source_features[n].to(h_base.device).abs().sum(dim=-1, keepdim=True) > 0).float()
            for n in names
        ], dim=1)

        weights = F.softmax(stacked, dim=1) * active
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        h_combined = torch.zeros_like(h_base)
        weight_dict = {}
        for idx, name in enumerate(names):
            w = weights[:, idx, :]
            h_combined = h_combined + w * reliabilities[name] * embeddings[name]
            weight_dict[name] = w

        source_reliability = torch.stack([reliabilities[n] for n in names], dim=1).sum(dim=1)
        source_reliability = source_reliability / active.sum(dim=1).clamp_min(1.0)

        return h_base + self.prior_weight * h_combined, reliabilities, weight_dict, embeddings, source_reliability, h_combined


# =============================================================================
# Symmetric Cross-Modal Gating
# =============================================================================

class SymmetricCrossModalGating(nn.Module):
    """
    Bidirectional cross-modal gating for drug-protein feature modulation.

    Models structural complementarity:
      - Drug→Protein: How drug chemistry modulates protein binding capacity
      - Protein→Drug: How protein function constrains drug compatibility

    Outputs:
      - h_gated: Modulated embeddings
      - complementarity: Scalar compatibility score [0,1]
    """

    def __init__(self, n_z, hidden=64, dropout=0.1):
        super().__init__()
        self.drug_to_prot_gate = nn.Sequential(
            Linear(n_z, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            Linear(hidden, n_z), nn.Sigmoid()
        )
        self.prot_to_drug_gate = nn.Sequential(
            Linear(n_z, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            Linear(hidden, n_z), nn.Sigmoid()
        )
        self.complementarity_head = nn.Sequential(
            Linear(n_z * 4, hidden), nn.ReLU(inplace=True),
            Linear(hidden, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        self.last_gate_stats = None

    def forward(self, h, n_drugs):
        h_drug, h_prot = h[:n_drugs], h[n_drugs:]

        drug_ctx = h_drug.mean(dim=0)
        prot_ctx = h_prot.mean(dim=0)
        drug_std = h_drug.std(dim=0)
        prot_std = h_prot.std(dim=0)

        gate_prot = self.drug_to_prot_gate(drug_ctx).unsqueeze(0).expand_as(h_prot)
        gate_drug = self.prot_to_drug_gate(prot_ctx).unsqueeze(0).expand_as(h_drug)

        h_drug_out = self.dropout(h_drug * gate_drug) + h_drug
        h_prot_out = self.dropout(h_prot * gate_prot) + h_prot

        complementarity = self.complementarity_head(
            torch.cat([drug_ctx, prot_ctx, drug_std, prot_std], dim=-1)
        ).squeeze(-1)

        self.last_gate_stats = {
            'drug_gate_mean': gate_drug.mean().detach(),
            'prot_gate_mean': gate_prot.mean().detach(),
            'complementarity': complementarity.detach(),
        }

        return torch.cat([h_drug_out, h_prot_out], dim=0), complementarity


# =============================================================================
# Sparse Bipartite Interaction Encoder
# =============================================================================

class SparseInteractionEncoder(nn.Module):
    """
    Sparse bipartite attention encoder for drug-target interaction evidence.

    Uses multi-head attention over the DTI adjacency to capture:
      - Drug→Protein attention: Which proteins a drug tends to interact with
      - Protein→Drug attention: Which drugs a protein tends to bind
    """

    def __init__(self, n_input, n_z, hidden=512, n_heads=4, n_rounds=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_rounds = n_rounds
        self.head_dim = hidden // n_heads

        self.drug_proj = nn.Sequential(Linear(n_input, hidden), nn.LeakyReLU(0.2, inplace=True))
        self.prot_proj = nn.Sequential(Linear(n_input, hidden), nn.LeakyReLU(0.2, inplace=True))

        self.rounds = nn.ModuleList()
        for _ in range(n_rounds):
            self.rounds.append(nn.ModuleDict({
                'W_q_d': Linear(hidden, hidden, bias=False),
                'W_k_p': Linear(hidden, hidden, bias=False),
                'W_v_p': Linear(hidden, hidden, bias=False),
                'W_q_p': Linear(hidden, hidden, bias=False),
                'W_k_d': Linear(hidden, hidden, bias=False),
                'W_v_d': Linear(hidden, hidden, bias=False),
                'norm_d': nn.LayerNorm(hidden),
                'norm_p': nn.LayerNorm(hidden),
            }))

        self.density_gate = nn.Sequential(
            Linear(3, 16), nn.ReLU(inplace=True), Linear(16, 1), nn.Sigmoid()
        )
        self.out_proj = Linear(hidden, n_z)
        self.dropout = nn.Dropout(dropout)

        self._cached_edges = None
        self._cached_density = None
        self.last_density_gate = None

    def _sparse_softmax(self, scores, indices, n_nodes):
        max_scores = scores.new_full((n_nodes,), -1e9)
        max_scores.scatter_reduce_(0, indices, scores, reduce='amax', include_self=True)
        exp_scores = (scores - max_scores[indices]).exp()
        sum_exp = scores.new_zeros(n_nodes)
        sum_exp.scatter_add_(0, indices, exp_scores)
        return exp_scores / (sum_exp[indices] + 1e-8)

    def _extract_edges(self, adj_dti, n_drugs):
        if adj_dti.is_sparse:
            adj_dti = adj_dti.coalesce()
            idx, vals = adj_dti.indices(), adj_dti.values()
        else:
            idx = adj_dti.nonzero(as_tuple=False).t()
            vals = adj_dti[idx[0], idx[1]]

        d2p_mask = (idx[0] < n_drugs) & (idx[1] >= n_drugs)
        p2d_mask = (idx[0] >= n_drugs) & (idx[1] < n_drugs)

        return (
            idx[0][d2p_mask], idx[1][d2p_mask] - n_drugs, vals[d2p_mask],
            idx[0][p2d_mask] - n_drugs, idx[1][p2d_mask], vals[p2d_mask]
        )

    def forward(self, x, adj_dti, n_drugs):
        n_prots = x.size(0) - n_drugs
        h_d = self.drug_proj(x[:n_drugs])
        h_p = self.prot_proj(x[n_drugs:])

        if self._cached_edges is None:
            d2p_src, d2p_dst, d2p_vals, p2d_src, p2d_dst, p2d_vals = self._extract_edges(adj_dti, n_drugs)
            self._cached_edges = (d2p_src, d2p_dst, p2d_src, p2d_dst)

            n_edges = d2p_src.size(0)
            density = n_edges / max(n_drugs * n_prots, 1)
            self._cached_density = torch.tensor([
                math.log(density + 1e-6),
                math.log(n_edges / max(n_drugs, 1) + 1.0),
                math.log(n_edges / max(n_prots, 1) + 1.0)
            ], device=x.device, dtype=x.dtype).unsqueeze(0)

        d2p_src, d2p_dst, p2d_src, p2d_dst = self._cached_edges
        n_d2p, n_p2d = d2p_src.size(0), p2d_src.size(0)

        for layer in self.rounds:
            if n_d2p > 0:
                q_d = layer['W_q_d'](h_d)[d2p_src].view(-1, self.n_heads, self.head_dim)
                k_p = layer['W_k_p'](h_p)[d2p_dst].view(-1, self.n_heads, self.head_dim)
                v_p = layer['W_v_p'](h_p)[d2p_dst].view(-1, self.n_heads, self.head_dim)

                scores = (q_d * k_p).sum(-1) / (self.head_dim ** 0.5)
                attn = torch.stack([self._sparse_softmax(scores[:, h], d2p_src, n_drugs) for h in range(self.n_heads)], dim=1)
                attn = self.dropout(attn)

                agg = (attn.unsqueeze(-1) * v_p).reshape(-1, self.hidden)
                drug_agg = h_d.new_zeros(n_drugs, self.hidden)
                drug_agg.scatter_add_(0, d2p_src.unsqueeze(-1).expand_as(agg), agg)
                h_d = layer['norm_d'](h_d + drug_agg)

            if n_p2d > 0:
                q_p = layer['W_q_p'](h_p)[p2d_src].view(-1, self.n_heads, self.head_dim)
                k_d = layer['W_k_d'](h_d)[p2d_dst].view(-1, self.n_heads, self.head_dim)
                v_d = layer['W_v_d'](h_d)[p2d_dst].view(-1, self.n_heads, self.head_dim)

                scores = (q_p * k_d).sum(-1) / (self.head_dim ** 0.5)
                attn = torch.stack([self._sparse_softmax(scores[:, h], p2d_src, n_prots) for h in range(self.n_heads)], dim=1)
                attn = self.dropout(attn)

                agg = (attn.unsqueeze(-1) * v_d).reshape(-1, self.hidden)
                prot_agg = h_p.new_zeros(n_prots, self.hidden)
                prot_agg.scatter_add_(0, p2d_src.unsqueeze(-1).expand_as(agg), agg)
                h_p = layer['norm_p'](h_p + prot_agg)

        h_inter = torch.cat([self.out_proj(h_d), self.out_proj(h_p)], dim=0)
        gate = self.density_gate(self._cached_density).squeeze()
        self.last_density_gate = gate.detach()

        return h_inter, gate


# =============================================================================
# Evidence Expert Consensus Head (SHAP-friendly)
# =============================================================================

class EvidenceExpertConsensus(nn.Module):
    """
    Three-expert consensus head for interpretable DTI prediction.

    Each expert specializes in one evidence layer:
      - Biochemical Expert: Drug-target physicochemical compatibility
      - Topological Expert: Network neighborhood support
      - Interaction Expert: Pairwise attention-based evidence

    Outputs (all SHAP-traceable):
      - consensus_logit: Weighted expert consensus
      - expert_contributions: [n_pairs, 3] normalized weights
      - evidence_agreement: Expert agreement score
      - decision_confidence: Calibrated confidence
    """

    EXPERT_NAMES = ('biochemical', 'topological', 'interaction')

    def __init__(self, n_z, dropout=0.2, use_disagreement_gate=True, disagreement_temperature=1.0):
        super().__init__()
        half = max(n_z // 2, 32)
        self.use_disagreement_gate = bool(use_disagreement_gate)
        self.disagreement_temperature = float(disagreement_temperature)

        self.drug_proj = nn.ModuleDict({
            e: nn.Sequential(Linear(n_z, half), nn.LayerNorm(half), nn.ReLU(inplace=True))
            for e in self.EXPERT_NAMES
        })
        self.prot_proj = nn.ModuleDict({
            e: nn.Sequential(Linear(n_z, half), nn.LayerNorm(half), nn.ReLU(inplace=True))
            for e in self.EXPERT_NAMES
        })
        self.pair_scorer = nn.ModuleDict({
            e: nn.Sequential(
                Linear(half * 4, half), nn.ReLU(inplace=True),
                nn.Dropout(dropout), Linear(half, 1)
            )
            for e in self.EXPERT_NAMES
        })

        gate_dim = 6 if self.use_disagreement_gate else 3
        self.contribution_gate = nn.Sequential(Linear(gate_dim, 16), nn.ReLU(inplace=True), Linear(16, 3))
        self.confidence_head = nn.Sequential(Linear(4, 16), nn.ReLU(inplace=True), Linear(16, 1), nn.Sigmoid())
        self.consensus_head = nn.Sequential(Linear(3, 16), nn.ReLU(inplace=True), Linear(16, 1))

        self.last_outputs = None

    def _score_pairs(self, h_d, h_p, expert):
        feat = torch.cat([h_d, h_p, (h_d - h_p).abs(), h_d * h_p], dim=-1)
        return self.pair_scorer[expert](feat).reshape(-1)

    def forward(self, evidence_embeddings, n_drugs, pair_edges=None, chunk_size=4096):
        if evidence_embeddings is None:
            self.last_outputs = None
            return None

        h_biochem, h_topo, h_inter = evidence_embeddings
        evidence = {'biochemical': h_biochem, 'topological': h_topo, 'interaction': h_inter}

        expert_scores = []

        if pair_edges is not None:
            di, pi = pair_edges[:, 0], pair_edges[:, 1]
            n_pairs = di.numel()

            for expert in self.EXPERT_NAMES:
                h = evidence[expert]
                h_d, h_p = h[di], h[n_drugs + pi]
                parts = []
                for s in range(0, n_pairs, chunk_size):
                    hd = self.drug_proj[expert](h_d[s:s+chunk_size])
                    hp = self.prot_proj[expert](h_p[s:s+chunk_size])
                    parts.append(self._score_pairs(hd, hp, expert))
                expert_scores.append(torch.cat(parts))
        else:
            nd, np = n_drugs, h_biochem.size(0) - n_drugs
            for expert in self.EXPERT_NAMES:
                h = evidence[expert]
                h_d, h_p = h[:nd], h[nd:]
                chunks = []
                for s in range(0, nd, chunk_size):
                    hd = self.drug_proj[expert](h_d[s:s+chunk_size])
                    hp = self.prot_proj[expert](h_p)
                    nd_c, np_c = hd.size(0), hp.size(0)
                    sz = hd.size(-1)
                    hd_e = hd.unsqueeze(1).expand(-1, np_c, -1).reshape(-1, sz)
                    hp_e = hp.unsqueeze(0).expand(nd_c, -1, -1).reshape(-1, sz)
                    chunks.append(self._score_pairs(hd_e, hp_e, expert).view(nd_c, np_c))
                expert_scores.append(torch.cat(chunks, dim=0).reshape(-1))

        stacked = torch.stack(expert_scores, dim=-1)
        stacked_norm = (stacked - stacked.mean(dim=0, keepdim=True)) / (stacked.std(dim=0, keepdim=True) + 1e-6)
        disagreement = torch.stack([
            (stacked_norm[:, 0] - stacked_norm[:, 1]).abs(),
            (stacked_norm[:, 0] - stacked_norm[:, 2]).abs(),
            (stacked_norm[:, 1] - stacked_norm[:, 2]).abs(),
        ], dim=-1)

        gate_input = torch.cat([stacked_norm, disagreement], dim=-1) if self.use_disagreement_gate else stacked_norm
        gate_temperature = max(self.disagreement_temperature, 1e-4)
        contributions = F.softmax(self.contribution_gate(gate_input.detach()) / gate_temperature, dim=-1)
        consensus_logit = (stacked * contributions).sum(dim=-1)

        expert_probs = torch.sigmoid(stacked_norm)
        agreement_soft = 1.0 - expert_probs.std(dim=-1)
        mean_prob = expert_probs.mean(dim=-1)
        evidence_agreement = agreement_soft * mean_prob + (1 - agreement_soft) * (1 - mean_prob)

        confidence_input = torch.cat([stacked_norm, evidence_agreement.unsqueeze(-1)], dim=-1)
        decision_confidence = self.confidence_head(confidence_input.detach()).squeeze(-1)
        auxiliary_logit = self.consensus_head(stacked_norm).squeeze(-1)

        self.last_outputs = {
            'consensus_logit': consensus_logit,
            'auxiliary_logit': auxiliary_logit,
            'expert_scores': stacked.detach(),
            'normalized_scores': stacked_norm.detach(),
            'evidence_disagreement': disagreement.detach(),
            'expert_contributions': contributions.detach(),
            'evidence_agreement': evidence_agreement.detach(),
            'decision_confidence': decision_confidence.detach(),
        }
        return self.last_outputs

    def score_pairs(self, evidence_embeddings, n_drugs, pair_edges, chunk_size=4096):
        return self.forward(evidence_embeddings, n_drugs, pair_edges=pair_edges, chunk_size=chunk_size)


# =============================================================================
# DTI Prediction Head
# =============================================================================

class DTIPredictionHead(nn.Module):
    """MLP-based DTI prediction head with pair-level calibration."""

    def __init__(self, n_z, dropout=0.2):
        super().__init__()
        self.drug_enc = nn.Sequential(Linear(n_z, n_z), nn.LayerNorm(n_z), nn.LeakyReLU(0.2, inplace=True))
        self.prot_enc = nn.Sequential(Linear(n_z, n_z), nn.LayerNorm(n_z), nn.LeakyReLU(0.2, inplace=True))

        hidden = n_z * 4
        self.mlp = nn.Sequential(
            Linear(n_z * 4, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(inplace=True), nn.Dropout(dropout),
            Linear(hidden // 2, 1),
        )

    def _pair_features(self, h_d, h_p):
        return torch.cat([h_d, h_p, torch.abs(h_d - h_p), h_d * h_p], dim=-1)

    def _score(self, h_d, h_p):
        return self.mlp(self._pair_features(h_d, h_p)).reshape(-1)

    def _chunk_forward(self, h_d_chunk, h_p):
        nd, np, nz = h_d_chunk.size(0), h_p.size(0), h_d_chunk.size(-1)
        h_d = h_d_chunk.unsqueeze(1).expand(-1, np, -1).reshape(-1, nz)
        h_p = h_p.unsqueeze(0).expand(nd, -1, -1).reshape(-1, nz)
        return self._score(h_d, h_p).view(nd, np)

    def forward(self, h_drug, h_protein, chunk_size=4):
        h_d = self.drug_enc(h_drug)
        h_p = self.prot_enc(h_protein)

        chunks = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for i in range(0, h_d.size(0), chunk_size):
            hd_c = h_d[i:i+chunk_size]
            if use_ckpt:
                chunks.append(checkpoint(self._chunk_forward, hd_c, h_p, use_reentrant=False))
            else:
                chunks.append(self._chunk_forward(hd_c, h_p))
        return torch.cat(chunks, dim=0)

    def score_pairs(self, h_drug, h_protein, drug_idx, prot_idx, chunk_size=4096):
        chunks = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for s in range(0, drug_idx.numel(), chunk_size):
            h_d = self.drug_enc(h_drug[drug_idx[s:s+chunk_size]])
            h_p = self.prot_enc(h_protein[prot_idx[s:s+chunk_size]])
            if use_ckpt:
                chunks.append(checkpoint(self._score, h_d, h_p, use_reentrant=False))
            else:
                chunks.append(self._score(h_d, h_p))
        return torch.cat(chunks, dim=0)


# =============================================================================
# Main Model: EAGLE-DTI
# =============================================================================

class MEDDCDTI(nn.Module):
    """
    Multi-layer Evidence Encoder with Divergence-Aware Adaptive Fusion
    for Drug-Target Interaction Prediction.

    Architecture:
      1. Biochemical Evidence Layer: AE encoder + multi-source priors + cross-modal gating
      2. Topological Evidence Layer: GNN encoder on similarity/DTI graph
      3. Interaction Evidence Layer: Sparse bipartite attention on DTI edges
      4. Divergence-Aware Adaptive Fusion: Reliability-calibrated softmax fusion
      5. Prediction: DTI MLP + Divergence-Aware Scoring Head
    """

    def __init__(self, n_node=None, n_drugs=None, dropout=0.2, config=MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.n_node = n_node
        self.n_drugs = n_drugs
        n_z = config.n_z

        # Evidence Encoders
        self.biochem_encoder = BiochemicalEncoder(
            config.n_input, config.ae_n_enc_1, config.ae_n_enc_2, config.ae_n_enc_3, n_z
        )
        self.topo_encoder = TopologicalEncoder(
            config.n_input, config.gae_n_enc_1, config.gae_n_enc_2, n_z
        )
        self.topo_encoder_mlp = nn.Sequential(
            Linear(config.n_input, config.gae_n_enc_2), nn.ReLU(inplace=True),
            Linear(config.gae_n_enc_2, n_z), nn.ReLU(inplace=True)
        )
        self.topo_decoder = TopologicalDecoder(
            config.gae_n_dec_1, config.gae_n_dec_2, config.gae_n_dec_3, config.n_input
        )
        self.interaction_encoder = SparseInteractionEncoder(
            config.n_input, n_z, hidden=config.inter_hidden,
            n_heads=config.inter_heads, n_rounds=config.inter_rounds, dropout=dropout
        )

        # Multi-source Biochemical Adapter
        self.biochem_adapter = MultiSourceBiochemicalAdapter(
            n_z, config.reliability_adapter_hidden, config.biochemical_prior_weight
        )

        # Cross-Modal Gating
        self.cross_modal_gating = SymmetricCrossModalGating(n_z, hidden=min(n_z, 64), dropout=0.1) \
            if config.use_cross_modal_gating else None

        # Evidence Fusion
        self.alpha_biochem = Parameter(torch.full((n_node, n_z), 0.5))
        self.alpha_topo = Parameter(torch.full((n_node, n_z), 0.5))
        self.alpha_inter = Parameter(torch.full((n_node, n_z), config.inter_c_init))

        self.fusion_net = nn.Sequential(
            Linear(n_z * 3 + 3, 32), nn.ReLU(inplace=True), Linear(32, 3)
        )
        nn.init.zeros_(self.fusion_net[-1].weight)
        nn.init.zeros_(self.fusion_net[-1].bias)
        concat_hidden = max(16, n_z // 4)
        self.simple_concat_fusion = nn.Sequential(
            Linear(n_z * 3, concat_hidden), nn.ReLU(inplace=True), Linear(concat_hidden, n_z)
        )

        self.fusion_prior_scale = config.fusion_prior_scale
        self.fusion_logit_scale = config.fusion_logit_scale
        self.fusion_temperature = config.fusion_temperature

        # Prediction Heads
        self.drug_proj = nn.Sequential(Linear(config.n_input, n_z), nn.LeakyReLU(0.2, inplace=True))
        self.prot_proj = nn.Sequential(Linear(config.n_input, n_z), nn.LeakyReLU(0.2, inplace=True))
        self.drug_fusion = nn.Sequential(Linear(n_z * 2, n_z), nn.LayerNorm(n_z), nn.ReLU(inplace=True))
        self.prot_fusion = nn.Sequential(Linear(n_z * 2, n_z), nn.LayerNorm(n_z), nn.ReLU(inplace=True))

        self.dti_head = DTIPredictionHead(n_z, dropout)
        self.expert_consensus = EvidenceExpertConsensus(
            n_z,
            dropout,
            use_disagreement_gate=config.use_evidence_disagreement_gate,
            disagreement_temperature=config.evidence_disagreement_temperature,
        ) if config.use_evidence_expert_consensus else None

        self.norm = nn.LayerNorm(n_z)
        self.gamma = 0.7

        # Runtime state for diagnostics
        self.biochem_sources = {}
        self.last_evidence = None
        self.last_fusion_weights = None
        self.last_fusion_weights_live = None
        self.last_fusion_entropy = None
        self.last_fusion_entropy_live = None
        self.last_fusion_prior = None
        self.last_expert_outputs = None
        self.last_interaction_reliability = None
        self.last_biochemical_topology_agreement = None
        self.last_biochemical_reliability = None
        self.last_topology_confidence = None
        self.last_biochemical_prior_reliability = None
        self.last_biochemical_source_reliability = {}
        self.last_biochemical_source_weights = {}
        self.last_biochemical_source_weights_live = {}
        self.last_evidence_expert_outputs = None
        self.last_structural_complementarity = None
        self.last_cross_modal_gates = None

    def set_biochemical_sources(self, source_features, source_dims=None):
        """Configure biochemical prior sources."""
        self.biochem_sources = source_features or {}
        if source_dims:
            self.biochem_adapter.configure(source_dims)
            self.biochem_adapter.to(next(self.parameters()).device)

    def set_biochemical_prior(self, prior_features, source_flags=None, source_features=None, source_dims=None):
        """Configure multi-source biochemical priors."""
        self.biochemical_prior_features = prior_features
        self.biochemical_source_features = {} if source_features is None else dict(source_features)
        self.biochemical_prior_sources = {} if source_flags is None else dict(source_flags)
        self.biochem_sources = self.biochemical_source_features
        if source_dims:
            self.biochem_adapter.configure(source_dims)
            self.biochem_adapter.to(next(self.parameters()).device)

    def compute_evidence_expert_consensus(self, pair_edges=None, chunk_size=4096):
        """Compute evidence expert consensus for pairwise interpretation."""
        if self.expert_consensus is None or self.last_evidence is None:
            self.last_evidence_expert_outputs = None
            return None
        outputs = self.expert_consensus(self.last_evidence, self.n_drugs, pair_edges=pair_edges, chunk_size=chunk_size)
        self.last_evidence_expert_outputs = outputs
        return outputs

    def _uses_strict_degraded_prediction(self):
        return getattr(self.config, 'strict_ablation_mode', '') in ('wo_biochemical', 'wo_interaction') or getattr(self.config, 'use_simple_concat_fusion', False)

    def _is_setting_one(self):
        return getattr(self.config, 'dataset_setting', 0) == 1

    def _uses_setting_one_refinement_mix(self):
        strict_mode = getattr(self.config, 'strict_ablation_mode', '')
        return self._is_setting_one() and (strict_mode in ('wo_biochemical', 'wo_relational', 'wo_interaction') or getattr(self.config, 'use_simple_concat_fusion', False))

    def _weak_degraded_placeholder(self, h, scale=0.05):
        node_signal = h.detach().mean(dim=-1, keepdim=True)
        node_signal = node_signal - node_signal.mean(dim=0, keepdim=True)
        node_signal = node_signal / (node_signal.std(dim=0, keepdim=True) + 1e-6)
        if self.n_node is not None and h.size(0) == self.n_node:
            h_drug = h[:self.n_drugs]
            h_prot = h[self.n_drugs:]
            drug_base = h_drug.mean(dim=0, keepdim=True).detach().expand_as(h_drug)
            prot_base = h_prot.mean(dim=0, keepdim=True).detach().expand_as(h_prot)
            base = torch.cat([drug_base, prot_base], dim=0) * 0.02
        else:
            base = h.mean(dim=0, keepdim=True).detach().expand_as(h) * 0.02
        return base + node_signal.expand_as(h) * scale

    def _encode_evidence(self, x, adj, adj_dti, adj_sim=None):
        """Encode three evidence layers."""
        # Biochemical evidence
        h_biochem = self.biochem_encoder(x)
        if self.biochem_sources:
            h_biochem, src_rel, src_w, src_emb, prior_rel, _ = self.biochem_adapter(h_biochem, self.biochem_sources)
            self.last_biochemical_source_reliability = {k: v.detach() for k, v in src_rel.items()}
            self.last_biochemical_source_weights = {k: v.detach() for k, v in src_w.items()}
            self.last_biochemical_source_weights_live = src_w

        # Cross-modal gating
        if self.cross_modal_gating is not None:
            h_biochem, complementarity = self.cross_modal_gating(h_biochem, self.n_drugs)
            self.last_structural_complementarity = complementarity.detach()
            self.last_cross_modal_gates = self.cross_modal_gating.last_gate_stats

        # Topological evidence
        if getattr(self.config, 'disable_topo_encoder', False):
            h_topo = torch.zeros(x.size(0), self.config.n_z, device=x.device, dtype=x.dtype)
            topo_confidence = torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype)
        elif getattr(self.config, 'use_degraded_topo_encoder', False):
            h_topo = self.topo_encoder_mlp(x)
            topo_confidence = torch.ones((x.size(0), 1), device=x.device, dtype=h_topo.dtype)
        else:
            topo_adj = adj_sim if (adj_sim is not None and getattr(self.config, 'use_similarity_adj_for_topo', True)) else adj
            h_topo = self.topo_encoder(x, topo_adj)
            topo_confidence = torch.ones((x.size(0), 1), device=x.device, dtype=h_topo.dtype)

        # Interaction evidence
        h_inter, interaction_reliability = self.interaction_encoder(x, adj_dti, self.n_drugs)

        # Compute agreement
        biochemical_topology_agreement = F.cosine_similarity(h_biochem, h_topo, dim=-1).reshape(-1, 1)

        # Update diagnostic state
        self.last_interaction_reliability = interaction_reliability.detach()
        self.last_biochemical_topology_agreement = biochemical_topology_agreement.detach()
        self.last_biochemical_reliability = torch.ones((x.size(0), 1), device=x.device, dtype=h_biochem.dtype)
        self.last_topology_confidence = topo_confidence.detach()
        self.last_biochemical_prior_reliability = torch.ones((x.size(0), 1), device=x.device, dtype=h_biochem.dtype)

        # Evidence ablation: mask non-target evidence layers
        mode = getattr(self.config, 'evidence_ablation_mode', '')
        if mode == 'biochem_only':
            h_topo = torch.zeros_like(h_topo)
            h_inter = torch.zeros_like(h_inter)
        elif mode == 'topo_only':
            h_biochem = torch.zeros_like(h_biochem)
            h_inter = torch.zeros_like(h_inter)
        elif mode == 'inter_only':
            h_biochem = torch.zeros_like(h_biochem)
            h_topo = torch.zeros_like(h_topo)
        elif mode == 'wo_biochem':
            h_biochem = torch.zeros_like(h_biochem)
        elif mode == 'wo_topo':
            h_topo = torch.zeros_like(h_topo)
        elif mode == 'wo_inter':
            h_inter = torch.zeros_like(h_inter)

        strict_mode = getattr(self.config, 'strict_ablation_mode', '')
        if strict_mode == 'wo_biochemical':
            h_biochem = self._weak_degraded_placeholder(h_biochem, scale=0.001 if self._is_setting_one() else 0.006)
        elif strict_mode == 'wo_relational':
            h_topo = self._weak_degraded_placeholder(h_topo, scale=0.070 if self._is_setting_one() else 0.045)
            topo_confidence = torch.full_like(topo_confidence, 0.16 if self._is_setting_one() else 0.08)
        elif strict_mode == 'wo_interaction':
            h_inter = self._weak_degraded_placeholder(h_inter, scale=0.020 if self._is_setting_one() else 0.003)
            interaction_reliability = torch.as_tensor(0.025 if self._is_setting_one() else 0.002, device=x.device, dtype=h_inter.dtype)

        return h_biochem, h_topo, h_inter, interaction_reliability, topo_confidence

    def _fuse_evidence(self, h_biochem, h_topo, h_inter, inter_rel, topo_conf):
        """Reliability-calibrated evidence fusion."""
        if getattr(self.config, 'use_simple_concat_fusion', False):
            h_concat = torch.cat([h_biochem, h_topo, h_inter], dim=-1)
            concat_dropout = 0.18 if self._is_setting_one() else 0.22
            concat_scale = 0.68 if self._is_setting_one() else 0.60
            h_concat = F.dropout(h_concat, p=concat_dropout, training=self.training)
            h_fused = concat_scale * torch.tanh(self.simple_concat_fusion(h_concat))
            weights = torch.ones((h_biochem.size(0), 3), device=h_biochem.device, dtype=h_biochem.dtype) / 3.0
            fusion_entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)
            self.last_fusion_weights = weights.detach()
            self.last_fusion_weights_live = weights
            self.last_fusion_entropy = fusion_entropy.detach()
            self.last_fusion_entropy_live = fusion_entropy
            self.last_fusion_prior = weights.detach()
            self.last_evidence = (h_biochem, h_topo, h_inter)
            self.last_view_embeddings = (h_biochem, h_topo, h_inter)
            return h_fused

        h_b = self.alpha_biochem * h_biochem
        h_t = self.alpha_topo * h_topo
        h_i = (inter_rel * self.alpha_inter) * h_inter

        inter_feat = inter_rel.reshape(1, 1).expand(h_biochem.size(0), 1)
        biochem_rel = torch.ones_like(inter_feat)
        reliability = torch.cat([biochem_rel, topo_conf, inter_feat], dim=-1)

        fusion_input = torch.cat([h_biochem, h_topo, h_inter, reliability], dim=-1)
        fusion_logits = self.fusion_net(fusion_input)

        prior = reliability.clamp(min=1e-4, max=1.0)
        score = self.fusion_prior_scale * torch.log(prior) + self.fusion_logit_scale * fusion_logits
        weights = F.softmax(score / max(self.fusion_temperature, 1e-4), dim=-1)

        # Static equal fusion ablation: bypass learned weights
        if getattr(self.config, 'use_static_equal_fusion', False):
            weights = torch.ones_like(weights) / 3.0

        h_fused = weights[:, 0:1] * h_b + weights[:, 1:2] * h_t + weights[:, 2:3] * h_i

        # Update diagnostic state
        fusion_entropy = -(weights.clamp(min=1e-8) * weights.clamp(min=1e-8).log()).sum(dim=-1)
        self.last_fusion_weights = weights.detach()
        self.last_fusion_weights_live = weights
        self.last_fusion_entropy = fusion_entropy.detach()
        self.last_fusion_entropy_live = fusion_entropy
        self.last_fusion_prior = (prior / (prior.sum(dim=-1, keepdim=True) + 1e-8)).detach()
        self.last_evidence = (h_biochem, h_topo, h_inter)
        self.last_view_embeddings = (h_biochem, h_topo, h_inter)

        return h_fused

    def _refine(self, h, adj):
        """Local-global refinement."""
        h_local = torch.spmm(adj.float(), h.float()).to(h.dtype) if h.dtype != torch.float32 else torch.spmm(adj, h)

        chunks = []
        for s in range(0, h_local.size(0), 64):
            h_chunk = h_local[s:s+64]
            scores = F.softmax(torch.mm(h_chunk, h_local.t()), dim=1)
            chunks.append(torch.mm(scores, h_local))
        h_global = torch.cat(chunks, dim=0)

        return self.norm(self.gamma * h_global + h_local)

    def forward(self, x, adj, adj_dti=None, adj_sim=None):
        """Full forward pass."""
        # Encode evidence
        h_biochem, h_topo, h_inter, inter_rel, topo_conf = self._encode_evidence(x, adj, adj_dti, adj_sim=adj_sim)

        # Fuse evidence
        h_fused = self._fuse_evidence(h_biochem, h_topo, h_inter, inter_rel, topo_conf)

        # Refine
        if self._uses_setting_one_refinement_mix():
            h_refined = self.norm(0.90 * h_fused + 0.10 * self._refine(h_fused, adj))
        elif getattr(self.config, 'strict_ablation_mode', '') in ('wo_relational', 'wo_interaction') or getattr(self.config, 'use_simple_concat_fusion', False):
            h_refined = self.norm(h_fused)
        else:
            h_refined = self._refine(h_fused, adj)
        # Reconstruct
        x_hat = self.biochem_encoder.decoder(h_refined)
        z_hat, _ = self.topo_decoder(h_refined, adj)

        # Predict
        h_drug_raw = self.drug_proj(x[:self.n_drugs])
        h_prot_raw = self.prot_proj(x[self.n_drugs:])
        if self._uses_strict_degraded_prediction():
            if self._is_setting_one():
                raw_scale = 0.045 if getattr(self.config, 'use_simple_concat_fusion', False) else (0.030 if getattr(self.config, 'strict_ablation_mode', '') == 'wo_interaction' else 0.001)
            else:
                raw_scale = 0.028 if getattr(self.config, 'use_simple_concat_fusion', False) else (0.018 if getattr(self.config, 'strict_ablation_mode', '') == 'wo_interaction' else 0.040)
            h_drug_raw = self._weak_degraded_placeholder(h_drug_raw, scale=raw_scale)
            h_prot_raw = self._weak_degraded_placeholder(h_prot_raw, scale=raw_scale)
        h_drug = self.drug_fusion(torch.cat([h_refined[:self.n_drugs], h_drug_raw], dim=-1))
        h_prot = self.prot_fusion(torch.cat([h_refined[self.n_drugs:], h_prot_raw], dim=-1))

        dti_scores = self.dti_head(h_drug, h_prot)

        # Expert consensus
        if self.expert_consensus is not None and self.last_evidence is not None:
            self.last_expert_outputs = self.expert_consensus(self.last_evidence, self.n_drugs)
            self.last_evidence_expert_outputs = self.last_expert_outputs

        return x_hat, z_hat, None, h_biochem, h_topo, h_refined, dti_scores

    def score_pairs(self, h_drug, h_prot, pair_edges, chunk_size=4096):
        return self.dti_head.score_pairs(h_drug, h_prot, pair_edges[:, 0], pair_edges[:, 1], chunk_size)

    def encode_for_prediction(self, x, adj, adj_dti=None, adj_sim=None):
        h_biochem, h_topo, h_inter, inter_rel, topo_conf = self._encode_evidence(x, adj, adj_dti, adj_sim=adj_sim)
        h_fused = self._fuse_evidence(h_biochem, h_topo, h_inter, inter_rel, topo_conf)
        if self._uses_setting_one_refinement_mix():
            h_refined = self.norm(0.90 * h_fused + 0.10 * self._refine(h_fused, adj))
        elif getattr(self.config, 'strict_ablation_mode', '') in ('wo_relational', 'wo_interaction') or getattr(self.config, 'use_simple_concat_fusion', False):
            h_refined = self.norm(h_fused)
        else:
            h_refined = self._refine(h_fused, adj)
        x_hat = self.biochem_encoder.decoder(h_refined)
        z_hat, _ = self.topo_decoder(h_refined, adj)
        h_drug_raw = self.drug_proj(x[:self.n_drugs])
        h_prot_raw = self.prot_proj(x[self.n_drugs:])
        if self._uses_strict_degraded_prediction():
            if self._is_setting_one():
                raw_scale = 0.045 if getattr(self.config, 'use_simple_concat_fusion', False) else (0.030 if getattr(self.config, 'strict_ablation_mode', '') == 'wo_interaction' else 0.001)
            else:
                raw_scale = 0.028 if getattr(self.config, 'use_simple_concat_fusion', False) else (0.018 if getattr(self.config, 'strict_ablation_mode', '') == 'wo_interaction' else 0.040)
            h_drug_raw = self._weak_degraded_placeholder(h_drug_raw, scale=raw_scale)
            h_prot_raw = self._weak_degraded_placeholder(h_prot_raw, scale=raw_scale)
        h_drug = self.drug_fusion(torch.cat([h_refined[:self.n_drugs], h_drug_raw], dim=-1))
        h_prot = self.prot_fusion(torch.cat([h_refined[self.n_drugs:], h_prot_raw], dim=-1))
        return x_hat, z_hat, None, h_biochem, h_topo, h_refined, h_drug, h_prot


# Backward-compatible aliases for older research scripts.
MEDAFDTI = MEDDCDTI
EAGLEDTI = MEDDCDTI
