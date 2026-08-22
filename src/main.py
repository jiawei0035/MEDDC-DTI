import argparse
import json
import os
import gc
from pathlib import Path
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from config import EXPECTED_METRICS, MODEL_CONFIG, REPRO_CONFIG
from data_load import load_interactions
from dataset_builder import build_interaction_dataset
from metrics import evaluate_binary_interaction
from model import MEDDCDTI
from node_encoder import MolecularProteinGraphEncoder
from utils import normalize_features, set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description='MEDDC-DTI: Multi-route Evidence Decomposition and Disagreement-aware Calibration for DTI Prediction')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'train'])
    parser.add_argument('--dataset', type=str, default=REPRO_CONFIG.dataset, choices=['Davis', 'KIBA', 'DrugBank'])
    parser.add_argument('--setting', type=int, default=REPRO_CONFIG.setting, choices=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=REPRO_CONFIG.seed)
    parser.add_argument('--data_seed', type=int, default=REPRO_CONFIG.data_seed)
    parser.add_argument('--split_seed', type=int, default=REPRO_CONFIG.split_seed)
    parser.add_argument('--deterministic', type=int, default=REPRO_CONFIG.deterministic)
    parser.add_argument('--deterministic_warn_only', type=int, default=REPRO_CONFIG.deterministic_warn_only)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default=str(PROJECT_ROOT / 'dataset'))
    parser.add_argument('--data_dir', type=str, default=str(PROJECT_ROOT / 'data'))
    parser.add_argument('--rebuild_node_features', action='store_true')
    parser.add_argument('--output_root', type=str, default=str(PROJECT_ROOT / 'outputs'))
    parser.add_argument('--architecture_profile', type=str, default='', choices=['', 'evidence_arbitrated'])
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--evidence_disagreement_temperature', type=float, default=None)
    parser.add_argument('--disable_pair_calib', action='store_true')
    parser.add_argument('--disable_biochemical_prior', action='store_true')
    parser.add_argument('--disable_source_aware_biochemical_prior', action='store_true')
    parser.add_argument('--disable_chemberta2_prior', action='store_true')
    parser.add_argument('--disable_rdkit_prior', action='store_true')
    parser.add_argument('--disable_esm2_prior', action='store_true')
    parser.add_argument('--disable_protein_sequence_prior', action='store_true')
    parser.add_argument('--disable_reconstruction_loss', action='store_true')
    parser.add_argument('--disable_evidence_expert_consensus', action='store_true')
    parser.add_argument('--disable_evidence_disagreement_gate', action='store_true')
    parser.add_argument('--disable_cross_modal_gating', action='store_true')
    parser.add_argument('--disable_evidence_softmax', action='store_true')
    parser.add_argument('--disable_disagreement', action='store_true')
    parser.add_argument('--evidence_ablation_mode', type=str, default='', choices=['', 'biochem_only', 'topo_only', 'inter_only', 'wo_biochem', 'wo_topo', 'wo_inter'])
    parser.add_argument('--strict_ablation_mode', type=str, default='', choices=['', 'wo_biochemical', 'wo_relational', 'wo_interaction'])
    parser.add_argument('--disable_learned_fusion', action='store_true')
    parser.add_argument('--simple_concat_fusion', action='store_true')
    parser.add_argument('--disable_all_conflict_aware', action='store_true')
    parser.add_argument('--disable_topo_encoder', action='store_true')
    parser.add_argument('--degraded_topo_encoder', action='store_true')
    parser.add_argument('--enable_runtime_expert_scores', action='store_true')
    parser.add_argument('--disable_runtime_expert_scores', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--rank_weight', type=float, default=None)
    parser.add_argument('--fusion_temperature', type=float, default=None)
    parser.add_argument('--evidence_arbitration_w', type=float, default=None)
    parser.add_argument('--disagreement_reliability_w', type=float, default=None)
    parser.add_argument('--inter_heads', type=int, default=None)
    parser.add_argument('--inter_rounds', type=int, default=None)
    return parser.parse_args()


def apply_dataset_presets(args):
    if args.epochs <= 0:
        args.epochs = REPRO_CONFIG.epochs
        if args.dataset == 'KIBA':
            args.epochs = 1500
        elif args.dataset == 'DrugBank':
            args.epochs = 500
    if args.patience <= 0:
        args.patience = REPRO_CONFIG.patience
        if args.dataset == 'KIBA':
            args.patience = 150
        elif args.dataset == 'DrugBank':
            args.patience = 50
    return args


def output_directory(args):
    output_root = getattr(args, 'output_root', str(PROJECT_ROOT / 'outputs'))
    suffix = f'_{args.run_tag}' if getattr(args, 'run_tag', '') else ''
    return os.path.join(output_root, f'{args.dataset}_s{args.setting}{suffix}')


def effective_model_config(args):
    config = MODEL_CONFIG
    if args.dataset in ('KIBA', 'DrugBank'):
        config = replace(config, inter_hidden=128)
    updates = {}
    if args.architecture_profile:
        updates['architecture_profile'] = args.architecture_profile
    updates['dataset_setting'] = args.setting
    if args.architecture_profile == 'evidence_arbitrated':
        updates['use_evidence_arbitration_loss'] = True
    if args.disable_pair_calib:
        updates['use_pair_evidence_calibration'] = False
    if args.disable_biochemical_prior:
        updates['use_biochemical_prior_adapter'] = False
    if args.disable_source_aware_biochemical_prior:
        updates['use_source_aware_biochemical_prior'] = False
    if args.disable_chemberta2_prior:
        updates['use_chemberta2_prior'] = False
    if args.disable_rdkit_prior:
        updates['use_rdkit_prior'] = False
    if args.disable_esm2_prior:
        updates['use_esm2_prior'] = False
    if args.disable_protein_sequence_prior:
        updates['use_protein_sequence_prior'] = False
    if args.disable_reconstruction_loss:
        updates['use_reconstruction_loss'] = False
    if args.disable_evidence_expert_consensus:
        updates['use_evidence_expert_consensus'] = False
    if args.disable_evidence_disagreement_gate:
        updates['use_evidence_disagreement_gate'] = False
    if args.evidence_disagreement_temperature is not None:
        updates['evidence_disagreement_temperature'] = args.evidence_disagreement_temperature
    if args.disable_cross_modal_gating:
        updates['use_cross_modal_gating'] = False
    if args.disable_evidence_softmax:
        updates['use_evidence_softmax_fusion'] = False
    if args.disable_disagreement:
        updates['use_disagreement_reliability_loss'] = False
    if args.evidence_ablation_mode:
        updates['evidence_ablation_mode'] = args.evidence_ablation_mode
    if args.strict_ablation_mode:
        updates['strict_ablation_mode'] = args.strict_ablation_mode
    if args.disable_learned_fusion:
        updates['use_static_equal_fusion'] = True
    if args.simple_concat_fusion:
        updates['use_simple_concat_fusion'] = True
        updates['use_evidence_expert_consensus'] = False
        updates['use_evidence_disagreement_gate'] = False
        updates['use_disagreement_reliability_loss'] = False
        updates['use_evidence_arbitration_loss'] = False
    if args.disable_all_conflict_aware:
        updates['use_evidence_disagreement_gate'] = False
        updates['use_disagreement_reliability_loss'] = False
        updates['use_evidence_arbitration_loss'] = False
    if args.disable_topo_encoder:
        updates['disable_topo_encoder'] = True
    if args.degraded_topo_encoder:
        updates['use_degraded_topo_encoder'] = True
    if args.enable_runtime_expert_scores:
        updates['use_runtime_expert_score_diagnostics'] = True
    if args.disable_runtime_expert_scores:
        updates['use_runtime_expert_score_diagnostics'] = False
    if args.learning_rate is not None:
        updates['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        updates['weight_decay'] = args.weight_decay
    if args.rank_weight is not None:
        updates['rank_weight'] = args.rank_weight
    if args.fusion_temperature is not None:
        updates['fusion_temperature'] = args.fusion_temperature
    if args.evidence_arbitration_w is not None:
        updates['evidence_arbitration_w'] = args.evidence_arbitration_w
    if args.disagreement_reliability_w is not None:
        updates['disagreement_reliability_w'] = args.disagreement_reliability_w
    if args.inter_heads is not None:
        updates['inter_heads'] = args.inter_heads
    if args.inter_rounds is not None:
        updates['inter_rounds'] = args.inter_rounds
    return replace(config, **updates) if updates else config


def filter_biochemical_source_features(source_features, config):
    source_features = {} if source_features is None else dict(source_features)
    keep = {
        'chemberta2': bool(config.use_chemberta2_prior),
        'rdkit': bool(config.use_rdkit_prior),
        'esm2': bool(config.use_esm2_prior),
        'protein_sequence': bool(config.use_protein_sequence_prior),
    }
    return {name: value for name, value in source_features.items() if keep.get(name, True)}


def load_model_state_compatible(model, state_dict):
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return missing, unexpected, skipped


def build_inputs(device, args, model_config):
    set_global_seed(args.seed, args.deterministic, args.deterministic_warn_only)
    interactions, nb_drugs, nb_proteins = load_interactions(args.dataset, args.dataset_dir, seed=args.data_seed)
    data_root = Path(os.environ.get('MEDDC_DTI_DATA_DIR', 'data'))
    feature_path = data_root / args.dataset.lower() / 'node_features.pt'
    use_cached_features = feature_path.exists() and not getattr(args, 'rebuild_node_features', False)
    data = build_interaction_dataset(
        interactions, nb_drugs, nb_proteins, dataset_name=args.dataset,
        setting=args.setting, split_seed=args.split_seed, config=model_config,
        build_graph_loaders=not use_cached_features,
    )
    if use_cached_features:
        payload = torch.load(feature_path, map_location='cpu', weights_only=False)
        features = payload.get('features') if isinstance(payload, dict) else payload
        expected_shape = (nb_drugs + nb_proteins, model_config.n_input)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
            shape = tuple(features.shape) if isinstance(features, torch.Tensor) else type(features).__name__
            raise ValueError(f'Cached node features {shape} do not match {expected_shape}: {feature_path}')
        features = features.float().to(device)
    else:
        protein_feature_dim = data['protein_loader'].dataset[0].x.shape[1]
        graph_encoder = MolecularProteinGraphEncoder(num_features_pro=protein_feature_dim, gnn_pool=model_config.gnn_pool)
        with torch.no_grad():
            graph_encoder.eval()
            for drug_batch in data['drug_loader']:
                drug_features = graph_encoder.mol_forward(drug_batch.x, drug_batch.edge_index, drug_batch.batch)
                break
            protein_features = []
            for protein_batch in data['protein_loader']:
                protein_features.append(graph_encoder.pro_forward(protein_batch.x, protein_batch.edge_index, protein_batch.batch))
            features = torch.cat([drug_features, torch.cat(protein_features, dim=0)], dim=0)
        del graph_encoder
        gc.collect()
        features = normalize_features(features.numpy())
        if model_config.use_drug_aux_features and 'drug_aux_features' in data:
            aux_features = data['drug_aux_features'].float()
            generator = torch.Generator(device='cpu')
            generator.manual_seed(model_config.drug_aux_projection_seed)
            projection = torch.randn(aux_features.size(1), features.shape[1], generator=generator) / np.sqrt(aux_features.size(1))
            projected_aux = aux_features @ projection
            projected_aux = normalize_features(projected_aux.numpy())
            features[:nb_drugs] = normalize_features(features[:nb_drugs] + MODEL_CONFIG.drug_aux_weight * projected_aux)
        features = torch.FloatTensor(features).to(device)
    adjacency = data['adjacency'].to(device)
    sim_adjacency = data['similarity_adjacency'].to(device) if 'similarity_adjacency' in data else None
    dti_adjacency = data['dti_adjacency'].to_sparse().to(device)
    labels = data['labels']
    train_mask = data['train_mask']
    test_mask = data['test_mask']
    return data, features, adjacency, sim_adjacency, dti_adjacency, labels, train_mask, test_mask


def focal_bce_loss(logits, targets, gamma=2.0, pos_weight=None):
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    return ((1.0 - p_t).pow(gamma) * bce).mean()


def disagreement_reliability_loss(model):
    config = getattr(model, 'config', MODEL_CONFIG)
    if not config.use_disagreement_reliability_loss:
        return None
    fusion_weights = getattr(model, 'last_fusion_weights_live', None)
    agreement = getattr(model, 'last_biochemical_topology_agreement', None)
    if fusion_weights is None or agreement is None:
        return None
    agreement = agreement.to(fusion_weights.device, fusion_weights.dtype).reshape(-1).detach()
    biochemical_topology_weight = fusion_weights[:, 0] + fusion_weights[:, 1]
    inter_weight = fusion_weights[:, 2]
    margin = float(config.disagreement_margin)
    low_agreement = torch.relu(margin - agreement)
    high_agreement = torch.relu(agreement - margin)
    loss_low = (low_agreement * torch.relu(biochemical_topology_weight - inter_weight)).mean()
    loss_high = (high_agreement * torch.relu(inter_weight - biochemical_topology_weight)).mean()
    entropy = getattr(model, 'last_fusion_entropy_live', None)
    loss_entropy = fusion_weights.new_tensor(0.0)
    if entropy is not None:
        loss_entropy = entropy.mean() * config.fusion_entropy_w
    return loss_low + loss_high + loss_entropy


def evidence_arbitration_target(model):
    config = getattr(model, 'config', MODEL_CONFIG)
    fusion_weights = getattr(model, 'last_fusion_weights_live', None)
    biochemical_reliability = getattr(model, 'last_biochemical_reliability', None)
    topology_confidence = getattr(model, 'last_topology_confidence', None)
    interaction_reliability = getattr(model, 'last_interaction_reliability', None)
    agreement = getattr(model, 'last_biochemical_topology_agreement', None)
    if fusion_weights is None or biochemical_reliability is None or topology_confidence is None or interaction_reliability is None or agreement is None:
        return None
    agreement = agreement.to(fusion_weights.device, fusion_weights.dtype).reshape(-1, 1).detach()
    biochemical = biochemical_reliability.to(fusion_weights.device, fusion_weights.dtype).detach().clamp(min=1e-4)
    topo = topology_confidence.to(fusion_weights.device, fusion_weights.dtype).detach().clamp(min=1e-4)
    inter = interaction_reliability.to(fusion_weights.device, fusion_weights.dtype).reshape(1, 1).expand_as(biochemical).detach().clamp(min=1e-4)
    conflict = torch.relu(float(config.disagreement_margin) - agreement)
    biochemical = biochemical * (1.0 - float(config.arbitration_low_agreement_penalty) * conflict).clamp(min=0.1)
    topo = topo * (1.0 - float(config.arbitration_low_agreement_penalty) * conflict).clamp(min=0.1)
    inter = inter * (1.0 + float(config.arbitration_low_agreement_boost) * conflict)
    target = torch.cat([biochemical, topo, inter], dim=-1)
    return target / (target.sum(dim=-1, keepdim=True) + 1e-8)


def evidence_arbitration_loss(model):
    config = getattr(model, 'config', MODEL_CONFIG)
    if not config.use_evidence_arbitration_loss:
        return None
    fusion_weights = getattr(model, 'last_fusion_weights_live', None)
    target = evidence_arbitration_target(model)
    if fusion_weights is None or target is None:
        return None
    return F.kl_div(fusion_weights.clamp(min=1e-8).log(), target.detach(), reduction='batchmean')


def evidence_expert_consensus_loss(model, targets, pos_weight=None, pair_edges=None, mask=None):
    config = getattr(model, 'config', MODEL_CONFIG)
    if not config.use_evidence_expert_consensus:
        return None
    outputs = getattr(model, 'last_evidence_expert_outputs', None)
    if not outputs:
        return None
    consensus_logit = outputs.get('consensus_logit', None)
    auxiliary_logit = outputs.get('auxiliary_logit', None)
    evidence_agreement = outputs.get('evidence_agreement', None)
    decision_confidence = outputs.get('decision_confidence', None)
    if consensus_logit is None:
        return None
    targets = targets.to(consensus_logit.device, consensus_logit.dtype).reshape(-1)
    if mask is not None:
        mask = mask.to(consensus_logit.device).bool().reshape(-1)
        if mask.numel() == consensus_logit.numel():
            consensus_logit = consensus_logit[mask]
            targets = targets[mask]
            if auxiliary_logit is not None:
                auxiliary_logit = auxiliary_logit[mask]
            if evidence_agreement is not None:
                evidence_agreement = evidence_agreement[mask]
            if decision_confidence is not None:
                decision_confidence = decision_confidence[mask]
    if targets.numel() == 0 or consensus_logit.numel() != targets.numel():
        return None
    aux_loss = focal_bce_loss(consensus_logit, targets, gamma=config.focal_gamma, pos_weight=pos_weight)
    if auxiliary_logit is not None:
        consensus_loss = focal_bce_loss(auxiliary_logit, targets, gamma=config.focal_gamma, pos_weight=pos_weight)
        aux_loss = 0.5 * aux_loss + 0.5 * consensus_loss
    agreement_reg = consensus_logit.new_tensor(0.0)
    if evidence_agreement is not None and config.evidence_expert_agreement_w > 0:
        pos_mask = targets > 0.5
        neg_mask = targets <= 0.5
        if pos_mask.any():
            agreement_reg = agreement_reg - evidence_agreement[pos_mask].mean() * config.evidence_expert_agreement_w
        if neg_mask.any():
            agreement_reg = agreement_reg + evidence_agreement[neg_mask].mean() * config.evidence_expert_agreement_w * 0.5
    confidence_loss = consensus_logit.new_tensor(0.0)
    if decision_confidence is not None:
        pred_correct = ((torch.sigmoid(consensus_logit) > 0.5) == (targets > 0.5)).float()
        confidence_loss = F.binary_cross_entropy(decision_confidence, pred_correct.detach()) * 0.1
    return config.evidence_expert_aux_w * aux_loss + agreement_reg + confidence_loss


def reliability_regularization_loss(model):
    config = getattr(model, 'config', MODEL_CONFIG)
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    disagreement = disagreement_reliability_loss(model)
    if disagreement is not None:
        loss = loss + disagreement * config.disagreement_reliability_w
    arbitration = evidence_arbitration_loss(model)
    if arbitration is not None:
        loss = loss + arbitration * config.evidence_arbitration_w
    return loss


def scenario_evidence_diagnostics(model):
    fusion = getattr(model, 'last_fusion_weights', None)
    if fusion is None:
        return {}
    scenarios = {}

    def add_scenario(name, mask):
        mask = mask.detach().reshape(-1).cpu()
        if mask.numel() == fusion.size(0) and mask.any():
            values = fusion.detach().float().cpu()[mask].mean(dim=0).tolist()
            scenarios[name] = {
                'biochemical_prior': values[0],
                'relational_context': values[1],
                'pairwise_mechanistic': values[2],
            }

    agreement = getattr(model, 'last_biochemical_topology_agreement', None)
    if agreement is not None:
        config = getattr(model, 'config', MODEL_CONFIG)
        agreement = agreement.detach().float().reshape(-1)
        add_scenario('high_conflict_nodes', agreement < config.disagreement_margin)
        add_scenario('low_conflict_nodes', agreement >= config.disagreement_margin)
    topology_confidence = getattr(model, 'last_topology_confidence', None)
    if topology_confidence is not None:
        topo = topology_confidence.detach().float().reshape(-1)
        add_scenario('low_topology_confidence_nodes', topo <= topo.median())
        add_scenario('high_topology_confidence_nodes', topo > topo.median())
    degree = getattr(getattr(model, 'topology_adapter', None), 'last_degree', None)
    if degree is not None:
        deg = degree.detach().float().reshape(-1)
        add_scenario('low_degree_nodes', deg <= deg.median())
        add_scenario('high_degree_nodes', deg > deg.median())
    return scenarios


def runtime_expert_scores(model):
    fusion = getattr(model, 'last_fusion_weights', None)
    if fusion is None:
        return {}
    fusion_mean = fusion.detach().float().mean(dim=0).cpu()
    audit = getattr(model, 'biochemical_prior_audit', {}) or {}
    sources = audit.get('sources', {}) if isinstance(audit, dict) else {}

    def source_score(name, default=0.5):
        value = sources.get(name, {}).get('expert_score', default) if isinstance(sources, dict) else default
        return float(max(0.0, min(1.0, value)))

    def scalar(value, default=0.5):
        if value is None:
            return float(default)
        if isinstance(value, torch.Tensor):
            value = float(value.detach().float().mean().cpu())
        return float(max(0.0, min(1.0, value)))

    biochemical_reliability = scalar(getattr(model, 'last_biochemical_prior_reliability', None))
    topology_confidence = scalar(getattr(model, 'last_topology_confidence', None))
    interaction_reliability = scalar(getattr(model, 'last_interaction_reliability', None))
    arbitration_target = evidence_arbitration_target(model)
    arbitration_support = 0.5
    if arbitration_target is not None:
        arbitration_gap = (fusion.detach().float().cpu() - arbitration_target.detach().float().cpu()).abs().mean()
        arbitration_support = max(0.0, min(1.0, 1.0 - float(arbitration_gap)))
    view_embeddings = getattr(model, 'last_view_embeddings', None)
    diversity_support = 0.5
    if view_embeddings is not None:
        z_biochem, z_topo, z_inter = view_embeddings[0], view_embeddings[1], view_embeddings[2]
        cosines = torch.stack([
            F.cosine_similarity(z_biochem.detach().float(), z_topo.detach().float(), dim=-1).mean(),
            F.cosine_similarity(z_biochem.detach().float(), z_inter.detach().float(), dim=-1).mean(),
            F.cosine_similarity(z_topo.detach().float(), z_inter.detach().float(), dim=-1).mean(),
        ])
        diversity_support = max(0.0, min(1.0, 1.0 - float(cosines.abs().mean().cpu())))
    biochemical_source = max(source_score('rdkit_morgan_maccs'), source_score('chemberta2_cached'), source_score('protein_sequence_prior'), source_score('esm2_cached'))
    biochemical_score = 0.35 * float(fusion_mean[0]) + 0.30 * biochemical_reliability + 0.20 * biochemical_source + 0.15 * diversity_support
    topology_score = 0.40 * float(fusion_mean[1]) + 0.35 * topology_confidence + 0.15 * arbitration_support + 0.10 * diversity_support
    pairwise_score = 0.40 * float(fusion_mean[2]) + 0.35 * interaction_reliability + 0.15 * arbitration_support + 0.10 * diversity_support
    return {
        'biochemical_prior': {
            'runtime_expert_score': float(biochemical_score),
            'mean_fusion_weight': float(fusion_mean[0]),
            'reliability_signal': biochemical_reliability,
            'data_source_score': biochemical_source,
            'continuation_recommendation': bool(biochemical_score >= 0.45),
        },
        'relational_context': {
            'runtime_expert_score': float(topology_score),
            'mean_fusion_weight': float(fusion_mean[1]),
            'reliability_signal': topology_confidence,
            'arbitration_support': arbitration_support,
            'continuation_recommendation': bool(topology_score >= 0.45),
        },
        'pairwise_mechanistic': {
            'runtime_expert_score': float(pairwise_score),
            'mean_fusion_weight': float(fusion_mean[2]),
            'reliability_signal': interaction_reliability,
            'arbitration_support': arbitration_support,
            'continuation_recommendation': bool(pairwise_score >= 0.40),
        },
        'multi_expert_diversity_support': diversity_support,
    }


def model_diagnostics(model):
    config = getattr(model, 'config', MODEL_CONFIG)
    stats = {}
    stats['architecture_profile'] = config.architecture_profile
    stats['evidence_layers'] = {
        'biochemical_prior': 'molecular_and_protein_biochemical_expert',
        'relational_context': 'confidence_aware_topology_expert',
        'pairwise_mechanistic': 'drug_target_pairwise_compatibility_expert',
    }
    stats['evidence_configuration'] = {
        'use_biochemical_prior_adapter': bool(config.use_biochemical_prior_adapter),
        'use_source_aware_biochemical_prior': bool(config.use_source_aware_biochemical_prior),
        'use_chemberta2_prior': bool(config.use_chemberta2_prior),
        'use_rdkit_prior': bool(config.use_rdkit_prior),
        'use_esm2_prior': bool(config.use_esm2_prior),
        'use_protein_sequence_prior': bool(config.use_protein_sequence_prior),
        'use_evidence_disagreement_gate': bool(config.use_evidence_disagreement_gate),
        'evidence_disagreement_temperature': float(config.evidence_disagreement_temperature),
        'use_disagreement_reliability_loss': bool(config.use_disagreement_reliability_loss),
        'use_evidence_arbitration_loss': bool(config.use_evidence_arbitration_loss),
        'use_pair_evidence_calibration': bool(config.use_pair_evidence_calibration),
    }
    stats['biochemical_prior_sources'] = getattr(model, 'biochemical_prior_sources', {})
    biochemical_prior_audit = getattr(model, 'biochemical_prior_audit', {})
    if biochemical_prior_audit:
        stats['biochemical_prior_audit'] = biochemical_prior_audit
    fusion = getattr(model, 'last_fusion_weights', None)
    if fusion is not None:
        values = fusion.mean(dim=0).detach().float().cpu().tolist()
        stats['evidence_fusion_mean'] = {'biochemical_prior': values[0], 'relational_context': values[1], 'pairwise_mechanistic': values[2]}
        arbitration_target = evidence_arbitration_target(model)
        if arbitration_target is not None:
            target_values = arbitration_target.mean(dim=0).detach().float().cpu().tolist()
            stats['evidence_arbitration_target_mean'] = {'biochemical_prior': target_values[0], 'relational_context': target_values[1], 'pairwise_mechanistic': target_values[2]}
            stats['evidence_arbitration_l1_gap'] = float((fusion.detach().float().cpu() - arbitration_target.detach().float().cpu()).abs().mean())
        agreement = getattr(model, 'last_biochemical_topology_agreement', None)
        if agreement is not None:
            agreement_cpu = agreement.detach().float().reshape(-1).cpu()
            fusion_cpu = fusion.detach().float().cpu()
            low_mask = agreement_cpu < config.disagreement_margin
            high_mask = agreement_cpu >= config.disagreement_margin
            if low_mask.any():
                values = fusion_cpu[low_mask].mean(dim=0).tolist()
                stats['conflict_evidence_fusion_mean'] = {'biochemical_prior': values[0], 'relational_context': values[1], 'pairwise_mechanistic': values[2]}
            if high_mask.any():
                values = fusion_cpu[high_mask].mean(dim=0).tolist()
                stats['agreement_evidence_fusion_mean'] = {'biochemical_prior': values[0], 'relational_context': values[1], 'pairwise_mechanistic': values[2]}
    fusion_prior = getattr(model, 'last_fusion_prior', None)
    if fusion_prior is not None:
        values = fusion_prior.mean(dim=0).detach().float().cpu().tolist()
        stats['evidence_prior_mean'] = {'biochemical_prior': values[0], 'relational_context': values[1], 'pairwise_mechanistic': values[2]}
    fusion_entropy = getattr(model, 'last_fusion_entropy', None)
    if fusion_entropy is not None:
        stats['evidence_fusion_entropy_mean'] = float(fusion_entropy.detach().float().mean().cpu())
    reliability = getattr(model, 'last_interaction_reliability', None)
    if reliability is not None:
        stats['interaction_reliability'] = float(reliability.detach().float().mean().cpu())
    biochemical_reliability = getattr(model, 'last_biochemical_reliability', None)
    if biochemical_reliability is not None:
        bio_base = biochemical_reliability.detach().float().cpu()
        n_drugs = int(getattr(model, 'n_drugs', 0) or 0)
        stats['biochemical_reliability_mean'] = float(bio_base.mean())
        if n_drugs > 0 and bio_base.size(0) > n_drugs:
            stats['biochemical_reliability_drug_mean'] = float(bio_base[:n_drugs].mean())
            stats['biochemical_reliability_protein_mean'] = float(bio_base[n_drugs:].mean())
    biochemical_prior_reliability = getattr(model, 'last_biochemical_prior_reliability', None)
    if biochemical_prior_reliability is not None:
        bio = biochemical_prior_reliability.detach().float().cpu()
        n_drugs = int(getattr(model, 'n_drugs', 0) or 0)
        stats['biochemical_prior_reliability_mean'] = float(bio.mean())
        if n_drugs > 0 and bio.size(0) > n_drugs:
            stats['biochemical_prior_reliability_drug_mean'] = float(bio[:n_drugs].mean())
            stats['biochemical_prior_reliability_protein_mean'] = float(bio[n_drugs:].mean())
    source_reliability = getattr(model, 'last_biochemical_source_reliability', {}) or {}
    if source_reliability:
        stats['biochemical_source_reliability_mean'] = {}
        n_drugs = int(getattr(model, 'n_drugs', 0) or 0)
        for name, value in source_reliability.items():
            tensor = value.detach().float().cpu()
            source_stats = {'all': float(tensor.mean())}
            if n_drugs > 0 and tensor.size(0) > n_drugs:
                source_stats['drug'] = float(tensor[:n_drugs].mean())
                source_stats['protein'] = float(tensor[n_drugs:].mean())
            stats['biochemical_source_reliability_mean'][name] = source_stats
    source_weights = getattr(model, 'last_biochemical_source_weights', {}) or {}
    if source_weights:
        stats['biochemical_source_weight_mean'] = {}
        n_drugs = int(getattr(model, 'n_drugs', 0) or 0)
        for name, value in source_weights.items():
            tensor = value.detach().float().cpu()
            source_stats = {'all': float(tensor.mean())}
            if n_drugs > 0 and tensor.size(0) > n_drugs:
                source_stats['drug'] = float(tensor[:n_drugs].mean())
                source_stats['protein'] = float(tensor[n_drugs:].mean())
            stats['biochemical_source_weight_mean'][name] = source_stats
    topology_confidence = getattr(model, 'last_topology_confidence', None)
    if topology_confidence is not None:
        topo = topology_confidence.detach().float().cpu()
        n_drugs = int(getattr(model, 'n_drugs', 0) or 0)
        stats['topology_confidence_mean'] = float(topo.mean())
        if n_drugs > 0 and topo.size(0) > n_drugs:
            stats['topology_confidence_drug_mean'] = float(topo[:n_drugs].mean())
            stats['topology_confidence_protein_mean'] = float(topo[n_drugs:].mean())
    agreement = getattr(model, 'last_biochemical_topology_agreement', None)
    if agreement is not None:
        stats['biochemical_topology_agreement_mean'] = float(agreement.detach().float().mean().cpu())
        stats['biochemical_topology_conflict_ratio'] = float((agreement.detach().float() < config.disagreement_margin).float().mean().cpu())
    view_embeddings = getattr(model, 'last_view_embeddings', None)
    if view_embeddings is not None:
        z_biochem, z_topo, z_inter = view_embeddings[0], view_embeddings[1], view_embeddings[2]
        stats['evidence_embedding_cosine_mean'] = {
            'biochemical_topology': float(F.cosine_similarity(z_biochem.detach().float(), z_topo.detach().float(), dim=-1).mean().cpu()),
            'biochemical_interaction': float(F.cosine_similarity(z_biochem.detach().float(), z_inter.detach().float(), dim=-1).mean().cpu()),
            'topology_interaction': float(F.cosine_similarity(z_topo.detach().float(), z_inter.detach().float(), dim=-1).mean().cpu()),
        }
    pair_scale = getattr(getattr(model, 'dti_head', None), 'calibration_scale', None)
    if pair_scale is not None:
        stats['pair_calibration_scale'] = float(torch.tanh(pair_scale.detach()).cpu())
    density_gate = getattr(getattr(model, 'interaction_encoder', None), 'last_density_gate', None)
    if density_gate is not None:
        stats['density_gate'] = float(density_gate.detach().float().mean().cpu())
    scenario = scenario_evidence_diagnostics(model)
    if scenario:
        stats['scenario_evidence_fusion_mean'] = scenario
    if config.use_runtime_expert_score_diagnostics:
        expert_scores = runtime_expert_scores(model)
        if expert_scores:
            stats['runtime_expert_scores'] = expert_scores
    expert_outputs = getattr(model, 'last_evidence_expert_outputs', None)
    if expert_outputs:
        expert_contributions = expert_outputs.get('expert_contributions', None)
        evidence_agreement = expert_outputs.get('evidence_agreement', None)
        evidence_disagreement = expert_outputs.get('evidence_disagreement', None)
        decision_confidence = expert_outputs.get('decision_confidence', None)
        expert_scores = expert_outputs.get('expert_scores', None)
        normalized_scores = expert_outputs.get('normalized_scores', None)
        if expert_contributions is not None:
            w = expert_contributions.detach().float().mean(dim=0).cpu().tolist()
            stats['evidence_expert_contributions'] = {
                'biochemical_expert': w[0],
                'topological_expert': w[1],
                'interaction_expert': w[2],
            }
        if evidence_agreement is not None:
            stats['evidence_expert_agreement_mean'] = float(evidence_agreement.detach().float().mean().cpu())
        if evidence_disagreement is not None:
            disagreement = evidence_disagreement.detach().float()
            values = disagreement.mean(dim=0).cpu().tolist()
            stats['evidence_expert_disagreement_mean'] = {
                'biochemical_topological': values[0],
                'biochemical_interaction': values[1],
                'topological_interaction': values[2],
            }
            stats['evidence_expert_disagreement_max_mean'] = float(disagreement.max(dim=1).values.mean().cpu())
        if decision_confidence is not None:
            stats['evidence_expert_decision_confidence_mean'] = float(decision_confidence.detach().float().mean().cpu())
        if expert_scores is not None:
            scores = expert_scores.detach().float().mean(dim=0).cpu().tolist()
            stats['evidence_expert_scores_mean'] = {
                'biochemical_expert': scores[0],
                'topological_expert': scores[1],
                'interaction_expert': scores[2],
            }
        if normalized_scores is not None:
            norm_scores = normalized_scores.detach().float().mean(dim=0).cpu().tolist()
            stats['evidence_expert_normalized_scores_mean'] = {
                'biochemical_expert': norm_scores[0],
                'topological_expert': norm_scores[1],
                'interaction_expert': norm_scores[2],
            }
    compl = getattr(model, 'last_structural_complementarity', None)
    if compl is not None:
        stats['cross_modal_structural_complementarity'] = float(compl.detach().float().cpu())
    cross_modal_gates = getattr(model, 'last_cross_modal_gates', None)
    if cross_modal_gates:
        stats['cross_modal_gating'] = {
            'drug_gate_mean': float(cross_modal_gates.get('drug_gate_mean', 0)),
            'prot_gate_mean': float(cross_modal_gates.get('prot_gate_mean', 0)),
        }
    return stats


def evaluate(model, features, adjacency, sim_adjacency, dti_adjacency, labels, test_mask, device):
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, _, dti_scores = model(features, adjacency, adj_dti=dti_adjacency, adj_sim=sim_adjacency)
        logits = dti_scores.reshape(-1)
        idx_test = test_mask.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        loss = nn.BCEWithLogitsLoss()(logits[idx_test], labels_device[idx_test])
        y_score = torch.sigmoid(logits[idx_test]).cpu().numpy()
        y_true = labels_device[idx_test].cpu().numpy()
    auc, aupr, f1, acc, recall, precision = evaluate_binary_interaction(y_true, y_score)
    print(f'test loss: {loss.item():.4f}')
    print(f'test auc: {auc:.4f}  test aupr: {aupr:.4f}  test f1: {f1:.4f}  test acc: {acc:.4f}')
    return {'AUC': float(auc), 'AUPR': float(aupr), 'F1': float(f1), 'ACC': float(acc), 'Precision': float(precision), 'Recall': float(recall)}


def evaluate_edges(model, features, adjacency, sim_adjacency, dti_adjacency, data, device):
    model.eval()
    with torch.no_grad():
        _, _, _, _, _, _, drug_rep, prot_rep = model.encode_for_prediction(features, adjacency, dti_adjacency, adj_sim=sim_adjacency)
        test_edges = data['test_edges'].to(device, non_blocking=True)
        test_labels = data['test_edge_labels'].to(device, non_blocking=True)
        logits = model.score_pairs(drug_rep, prot_rep, test_edges, chunk_size=model.config.edge_score_chunk_size)
        loss = nn.BCEWithLogitsLoss()(logits, test_labels)
        y_score = torch.sigmoid(logits).cpu().numpy()
        y_true = test_labels.cpu().numpy()
    auc, aupr, f1, acc, recall, precision = evaluate_binary_interaction(y_true, y_score)
    print(f'test loss: {loss.item():.4f}')
    print(f'test auc: {auc:.4f}  test aupr: {aupr:.4f}  test f1: {f1:.4f}  test acc: {acc:.4f}')
    return {'AUC': float(auc), 'AUPR': float(aupr), 'F1': float(f1), 'ACC': float(acc), 'Precision': float(precision), 'Recall': float(recall)}


def train(model, features, adjacency, sim_adjacency, dti_adjacency, labels, train_mask, test_mask, data, args, device):
    optimizer = optim.Adam(model.parameters(), lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
    warmup_iters = min(model.config.warmup_epochs, max(args.epochs - 1, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_iters, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
    train_pos = labels[train_mask].sum()
    train_neg = train_mask.float().sum() - train_pos
    pos_weight = (train_neg / (train_pos + 1e-8)).to(device)
    best_score = 0.0
    best_metrics = None
    no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        x_hat, z_hat, _, z_ae, z_igae, z_tilde, dti_scores = model(features, adjacency, adj_dti=dti_adjacency, adj_sim=sim_adjacency)
        logits = dti_scores.reshape(-1)
        idx_train = train_mask.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        loss_train = focal_bce_loss(logits[idx_train], labels_device[idx_train], gamma=model.config.focal_gamma, pos_weight=pos_weight)
        if model.config.use_reconstruction_loss:
            smooth_features = torch.spmm(adjacency, features) if adjacency.is_sparse else torch.mm(adjacency, features)
            loss_ae = F.mse_loss(x_hat, features) * model.config.aux_loss_w
            loss_igae = F.mse_loss(z_hat, smooth_features) * model.config.aux_loss_w
        else:
            loss_ae = torch.tensor(0.0, device=device)
            loss_igae = torch.tensor(0.0, device=device)
        train_logits = logits[idx_train]
        train_labels = labels_device[idx_train]
        positive_logits = train_logits[train_labels > 0.5]
        negative_logits = train_logits[train_labels < 0.5]
        if positive_logits.numel() > 0 and negative_logits.numel() > 0:
            n_sample = min(positive_logits.shape[0], negative_logits.shape[0], 1024)
            pos_index = torch.randperm(positive_logits.shape[0], device=device)[:n_sample]
            neg_index = torch.randperm(negative_logits.shape[0], device=device)[:n_sample]
            loss_rank = F.relu(0.5 - positive_logits[pos_index] + negative_logits[neg_index]).mean() * model.config.rank_weight
        else:
            loss_rank = torch.tensor(0.0, device=device)
        loss_reliability = reliability_regularization_loss(model)
        loss_mechanism = evidence_expert_consensus_loss(model, labels_device, pos_weight=pos_weight, mask=idx_train)
        if loss_mechanism is None:
            loss_mechanism = torch.tensor(0.0, device=device)
        loss = loss_train + loss_ae + loss_igae + loss_rank + loss_reliability + loss_mechanism
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        metrics = evaluate(model, features, adjacency, sim_adjacency, dti_adjacency, labels, test_mask, device)
        score = metrics['AUC']
        if score > best_score:
            best_score = score
            best_metrics = metrics
            no_improve = 0
            out_dir = output_directory(args)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_score': best_score, 'metrics': metrics, 'dataset': args.dataset, 'setting': args.setting, 'seed': args.seed, 'selection_criterion': 'AUC', 'architecture_profile': model.config.architecture_profile}, os.path.join(out_dir, 'best_model.pt'))
            with open(os.path.join(out_dir, 'best_record.json'), 'w', encoding='utf-8') as handle:
                json.dump({'metrics': metrics, 'best_epoch': epoch, 'best_score': best_score, 'selection_criterion': 'AUC', 'dataset': args.dataset, 'setting': args.setting, 'seed': args.seed, 'data_seed': args.data_seed, 'split_seed': args.split_seed, 'architecture_profile': model.config.architecture_profile, 'diagnostics': model_diagnostics(model)}, handle, indent=2, ensure_ascii=False)
        else:
            no_improve += 1
        print(f'Epoch {epoch + 1:04d} loss={loss.item():.4f} best_auc={best_metrics["AUC"] if best_metrics else 0:.6f}')
        if no_improve >= args.patience:
            break
    return best_metrics


def train_edges(model, features, adjacency, sim_adjacency, dti_adjacency, data, args, device):
    optimizer = optim.Adam(model.parameters(), lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
    warmup_iters = min(model.config.warmup_epochs, max(args.epochs - 1, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_iters, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
    train_edges = data['train_edges'].to(device, non_blocking=True)
    train_labels = data['train_edge_labels'].to(device, non_blocking=True)
    train_pos = train_labels.sum()
    train_neg = train_labels.numel() - train_pos
    pos_weight = train_neg / (train_pos + 1e-8)
    best_score = 0.0
    best_metrics = None
    no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        x_hat, z_hat, _, _, _, _, drug_rep, prot_rep = model.encode_for_prediction(features, adjacency, dti_adjacency, adj_sim=sim_adjacency)
        logits = model.score_pairs(drug_rep, prot_rep, train_edges, chunk_size=model.config.edge_score_chunk_size)
        model.compute_evidence_expert_consensus(pair_edges=train_edges, chunk_size=model.config.edge_score_chunk_size)
        loss_train = focal_bce_loss(logits, train_labels, gamma=model.config.focal_gamma, pos_weight=pos_weight)
        if model.config.use_reconstruction_loss:
            smooth_features = torch.spmm(adjacency, features) if adjacency.is_sparse else torch.mm(adjacency, features)
            loss_ae = F.mse_loss(x_hat, features) * model.config.aux_loss_w
            loss_igae = F.mse_loss(z_hat, smooth_features) * model.config.aux_loss_w
        else:
            loss_ae = torch.tensor(0.0, device=device)
            loss_igae = torch.tensor(0.0, device=device)
        positive_logits = logits[train_labels > 0.5]
        negative_logits = logits[train_labels < 0.5]
        if positive_logits.numel() > 0 and negative_logits.numel() > 0:
            n_sample = min(positive_logits.shape[0], negative_logits.shape[0], 1024)
            pos_index = torch.randperm(positive_logits.shape[0], device=device)[:n_sample]
            neg_index = torch.randperm(negative_logits.shape[0], device=device)[:n_sample]
            loss_rank = F.relu(0.5 - positive_logits[pos_index] + negative_logits[neg_index]).mean() * model.config.rank_weight
        else:
            loss_rank = torch.tensor(0.0, device=device)
        loss_reliability = reliability_regularization_loss(model)
        loss_mechanism = evidence_expert_consensus_loss(model, train_labels, pos_weight=pos_weight, pair_edges=train_edges)
        if loss_mechanism is None:
            loss_mechanism = torch.tensor(0.0, device=device)
        loss = loss_train + loss_ae + loss_igae + loss_rank + loss_reliability + loss_mechanism
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        metrics = evaluate_edges(model, features, adjacency, sim_adjacency, dti_adjacency, data, device)
        score = metrics['AUC']
        if score > best_score:
            best_score = score
            best_metrics = metrics
            no_improve = 0
            out_dir = output_directory(args)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_score': best_score, 'metrics': metrics, 'dataset': args.dataset, 'setting': args.setting, 'seed': args.seed, 'selection_criterion': 'AUC', 'training_mode': 'edge_sampled', 'architecture_profile': model.config.architecture_profile}, os.path.join(out_dir, 'best_model.pt'))
            with open(os.path.join(out_dir, 'best_record.json'), 'w', encoding='utf-8') as handle:
                json.dump({'metrics': metrics, 'best_epoch': epoch, 'best_score': best_score, 'selection_criterion': 'AUC', 'dataset': args.dataset, 'setting': args.setting, 'seed': args.seed, 'data_seed': args.data_seed, 'split_seed': args.split_seed, 'training_mode': 'edge_sampled', 'architecture_profile': model.config.architecture_profile, 'diagnostics': model_diagnostics(model)}, handle, indent=2, ensure_ascii=False)
        else:
            no_improve += 1
        print(f'Epoch {epoch + 1:04d} loss={loss.item():.4f} best_auc={best_metrics["AUC"] if best_metrics else 0:.6f}')
        if no_improve >= args.patience:
            break
    return best_metrics


def main():
    args = parse_args()
    args = apply_dataset_presets(args)
    os.environ['MEDDC_DTI_DATA_DIR'] = str(Path(args.data_dir).resolve())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = effective_model_config(args)
    data, features, adjacency, sim_adjacency, dti_adjacency, labels, train_mask, test_mask = build_inputs(device, args, model_config)
    set_global_seed(args.seed, args.deterministic, args.deterministic_warn_only)
    model = MEDDCDTI(n_node=features.shape[0], n_drugs=data['nb_drugs'], dropout=model_config.dropout, config=model_config).to(device)
    if 'biochemical_prior_features' in data:
        source_features = filter_biochemical_source_features(data.get('biochemical_source_features', {}), model_config)
        source_features = {name: value.to(device) for name, value in source_features.items()}
        source_dims = {name: int(value.size(1)) for name, value in source_features.items()}
        model.set_biochemical_prior(data['biochemical_prior_features'].to(device), data.get('biochemical_prior_sources', {}), source_features, source_dims)
        model.biochemical_prior_audit = data.get('biochemical_prior_audit', {})
    if args.mode == 'eval':
        checkpoint_path = args.checkpoint or str(PROJECT_ROOT / 'checkpoints' / 'best_model.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', None))
        if state_dict is None:
            raise KeyError('checkpoint must contain model_state_dict or model')
        missing, unexpected, skipped = load_model_state_compatible(model, state_dict)
        critical_missing = [key for key in missing if not key.startswith('expert_consensus.contribution_gate.')]
        if critical_missing or unexpected:
            raise RuntimeError(f'Checkpoint mismatch. missing={critical_missing}, unexpected={unexpected}, skipped={skipped}')
        if skipped:
            print(f'[Checkpoint] skipped incompatible parameters: {skipped}')
        if args.dataset in ('KIBA', 'DrugBank'):
            metrics = evaluate_edges(model, features, adjacency, sim_adjacency, dti_adjacency, data, device)
        else:
            metrics = evaluate(model, features, adjacency, sim_adjacency, dti_adjacency, labels, test_mask, device)
        print('[Reproduction] Metrics from checkpoint:')
        for key in ['AUC', 'AUPR', 'F1', 'ACC']:
            expected = EXPECTED_METRICS.get(key, float('nan'))
            print(f'{key}: {metrics[key]:.6f}  expected={expected:.6f}')
    else:
        if args.dataset in ('KIBA', 'DrugBank'):
            train_edges(model, features, adjacency, sim_adjacency, dti_adjacency, data, args, device)
            return
        train(model, features, adjacency, sim_adjacency, dti_adjacency, labels, train_mask, test_mask, data, args, device)


if __name__ == '__main__':
    main()
