"""MEDDC-DTI publication and reproducibility settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Reproducibility and training settings."""
    project_name: str = "MEDDC-DTI"
    paper_title: str = "Multi-route Evidence Decomposition and Disagreement-aware Calibration for Drug-Target Interaction Prediction"
    paper_title_cn: str = "基于多路证据分解与分歧感知校准的药物–靶点相互作用预测"
    dataset: str = "Davis"
    setting: int = 2
    seed: int = 3
    data_seed: int = 3
    split_seed: int = 3
    deterministic: int = 1
    deterministic_warn_only: int = 1
    epochs: int = 2000
    patience: int = 200


@dataclass(frozen=True)
class ModelConfig:
    """MEDDC-DTI model architecture configuration."""

    # Architecture profile
    architecture_profile: str = "evidence_arbitrated"

    # Embedding dimensions
    n_input: int = 256
    n_z: int = 256
    dropout: float = 0.2

    # Biochemical Encoder (AE)
    ae_n_enc_1: int = 128
    ae_n_enc_2: int = 256
    ae_n_enc_3: int = 512
    ae_n_dec_1: int = 512
    ae_n_dec_2: int = 256
    ae_n_dec_3: int = 128

    # Topological Encoder (GNN)
    gae_n_enc_1: int = 128
    gae_n_enc_2: int = 256
    gae_n_enc_3: int = 256
    gae_n_dec_1: int = 256
    gae_n_dec_2: int = 256
    gae_n_dec_3: int = 128

    # Interaction Encoder (Sparse Bipartite Attention)
    inter_hidden: int = 512
    inter_heads: int = 4
    inter_rounds: int = 2
    inter_c_init: float = 0.1
    use_density_gate: bool = True

    # Multi-Source Biochemical Prior
    use_biochemical_prior_adapter: bool = True
    use_source_aware_biochemical_prior: bool = True
    use_chemberta2_prior: bool = True
    use_rdkit_prior: bool = True
    use_esm2_prior: bool = True
    use_protein_sequence_prior: bool = True
    biochemical_prior_weight: float = 0.25
    reliability_adapter_hidden: int = 64
    chemberta2_cache_names: tuple = ("chemberta2_embeddings", "chemberta2")
    esm2_cache_names: tuple = ("esm2_embeddings", "esm2")

    # Cross-Modal Gating
    use_cross_modal_gating: bool = True

    # Evidence Fusion
    use_evidence_softmax_fusion: bool = True
    fusion_prior_scale: float = 1.0
    fusion_logit_scale: float = 0.5
    fusion_temperature: float = 0.75

    # Evidence Expert Consensus
    use_evidence_expert_consensus: bool = True
    use_evidence_disagreement_gate: bool = True
    evidence_disagreement_temperature: float = 1.0
    evidence_expert_aux_w: float = 0.01
    evidence_expert_agreement_w: float = 0.005

    # Regularization Losses
    use_disagreement_reliability_loss: bool = True
    disagreement_reliability_w: float = 0.02
    disagreement_margin: float = 0.15
    fusion_entropy_w: float = 0.002
    use_evidence_arbitration_loss: bool = True
    evidence_arbitration_w: float = 0.03

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 50
    focal_gamma: float = 2.0
    rank_weight: float = 0.1
    aux_loss_w: float = 0.01
    use_reconstruction_loss: bool = True

    # Edge sampling (for large datasets)
    edge_score_chunk_size: int = 4096
    drugbank_unobserved_neg_ratio: float = 1.0
    drugbank_unobserved_neg_seed: int = 2027

    # Diagnostics
    use_runtime_expert_score_diagnostics: bool = True

    # Protein graph
    esm2_contact_thresh: float = 0.2

    # Ablation controls
    evidence_ablation_mode: str = ''  # '', 'biochem_only', 'topo_only', 'inter_only', 'wo_biochem', 'wo_topo', 'wo_inter'
    strict_ablation_mode: str = ''
    dataset_setting: int = 0
    use_static_equal_fusion: bool = False
    use_simple_concat_fusion: bool = False
    disable_topo_encoder: bool = False
    use_degraded_topo_encoder: bool = False
    use_similarity_adj_for_topo: bool = True

    # Pairwise calibration
    use_pair_evidence_calibration: bool = True
    pair_calibration_init: float = 0.05
    arbitration_low_agreement_boost: float = 1.0
    arbitration_low_agreement_penalty: float = 0.5

    # Dataset builder compatibility
    gnn_pool: str = "mean"
    use_drug_aux_features: bool = False
    drug_aux_weight: float = 0.02
    drug_aux_projection_seed: int = 17
    use_drug_aux_similarity_graph: bool = False
    drug_aux_similarity_top_k: int = 8
    drug_aux_similarity_threshold: float = 0.25
    drug_aux_similarity_weight: float = 0.05
    use_cached_drug_embeddings: bool = True
    drug_embedding_top_k: int = 8
    drug_embedding_threshold: float = 0.2
    drug_embedding_graph_weight: float = 0.05
    use_cached_protein_embeddings: bool = True
    protein_embedding_top_k: int = 8
    protein_embedding_threshold: float = 0.2
    protein_embedding_graph_weight: float = 0.05
    biochemical_prior_projection_seed: int = 2029


REPRO_CONFIG = ReproducibilityConfig()
MODEL_CONFIG = ModelConfig()

EXPECTED_METRICS = {
    "AUC": 0.98158598400632,
    "AUPR": 0.9809061183220472,
    "F1": 0.937879810938555,
    "ACC": 0.9363101124763489,
}
