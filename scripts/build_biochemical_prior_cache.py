import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_load import load_interactions


def ordered_entities(interactions):
    drug_to_smiles = {}
    protein_to_sequence = {}
    for drug_id, protein_id, _, smiles, sequence in interactions:
        drug_to_smiles.setdefault(drug_id, smiles)
        protein_to_sequence.setdefault(protein_id, sequence)
    drug_ids = sorted(drug_to_smiles)
    protein_ids = sorted(protein_to_sequence)
    return drug_ids, [drug_to_smiles[item] for item in drug_ids], protein_ids, [protein_to_sequence[item] for item in protein_ids]


def load_embedding_map(path):
    if not path:
        return None
    if path.endswith('.pt'):
        payload = torch.load(path, map_location='cpu', weights_only=False)
    elif path.endswith('.npz'):
        payload = dict(np.load(path, allow_pickle=True))
    else:
        payload = np.load(path, allow_pickle=True).item()
    if isinstance(payload, dict) and 'embeddings' in payload:
        payload = payload['embeddings']
    return payload


def align_embedding_map(ids, embedding_map):
    if embedding_map is None:
        return None
    if isinstance(embedding_map, torch.Tensor):
        return embedding_map.float()
    if isinstance(embedding_map, np.ndarray):
        return torch.from_numpy(embedding_map.astype(np.float32)).float()
    return torch.stack([torch.as_tensor(embedding_map[item], dtype=torch.float32) for item in ids], dim=0)


def encode_texts_with_transformers(texts, model_name, batch_size, max_length, device, local_files_only=False):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0).float()


def load_esm2_pooled_cache(protein_ids, esm2_dir):
    if not esm2_dir or not os.path.isdir(esm2_dir):
        return None, {'available': False, 'coverage': 0.0, 'missing': len(protein_ids), 'truncated': 0}
    embeddings = []
    missing = []
    truncated = 0
    for protein_id in protein_ids:
        path = os.path.join(esm2_dir, protein_id + '.pt')
        if not os.path.exists(path):
            missing.append(protein_id)
            continue
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if 'pooled_emb' in payload:
            pooled = payload['pooled_emb'].float()
        else:
            pooled = payload['residue_emb'].float().mean(dim=0)
        embeddings.append(pooled)
        truncated += int(bool(payload.get('truncated', False)))
    if missing:
        return None, {'available': False, 'coverage': 1.0 - len(missing) / max(len(protein_ids), 1), 'missing': len(missing), 'truncated': truncated}
    return torch.stack(embeddings, dim=0).float(), {'available': True, 'coverage': 1.0, 'missing': 0, 'truncated': truncated}


def save_cache(output_dir, dataset, drug_embeddings, protein_embeddings):
    os.makedirs(output_dir, exist_ok=True)
    if drug_embeddings is not None:
        torch.save({'embeddings': drug_embeddings.float()}, os.path.join(output_dir, 'chemberta2_embeddings.pt'))
        torch.save({'embeddings': drug_embeddings.float()}, os.path.join(output_dir, f'{dataset.lower()}_chemberta2_embeddings.pt'))
    if protein_embeddings is not None:
        torch.save({'embeddings': protein_embeddings.float()}, os.path.join(output_dir, 'esm2_embeddings.pt'))
        torch.save({'embeddings': protein_embeddings.float()}, os.path.join(output_dir, f'{dataset.lower()}_esm2_embeddings.pt'))


def expert_source_score(coverage, performance_support, continuity, interpretability, cost):
    return float(
        0.35 * coverage
        + 0.25 * performance_support
        + 0.20 * continuity
        + 0.10 * interpretability
        + 0.10 * (1.0 - cost)
    )


def build_audit(dataset, drug_ids, protein_ids, drug_embeddings, protein_embeddings, esm2_status):
    sources = {}
    sources['rdkit_morgan_maccs'] = {
        'available': True,
        'coverage': 1.0,
        'supports_final_performance': 0.80,
        'continuity_potential': 0.90,
        'interpretability': 0.85,
        'computational_cost': 0.05,
        'expert_score': expert_source_score(1.0, 0.80, 0.90, 0.85, 0.05),
        'role': 'drug structural and physicochemical expert',
    }
    sources['protein_sequence_prior'] = {
        'available': True,
        'coverage': 1.0,
        'supports_final_performance': 0.70,
        'continuity_potential': 0.85,
        'interpretability': 0.75,
        'computational_cost': 0.03,
        'expert_score': expert_source_score(1.0, 0.70, 0.85, 0.75, 0.03),
        'role': 'lightweight protein biochemical statistics expert',
    }
    chemberta_available = drug_embeddings is not None
    sources['chemberta2_cached'] = {
        'available': bool(chemberta_available),
        'coverage': 1.0 if chemberta_available else 0.0,
        'supports_final_performance': 0.82,
        'continuity_potential': 0.92,
        'interpretability': 0.62,
        'computational_cost': 0.65 if chemberta_available else 1.0,
        'expert_score': expert_source_score(1.0 if chemberta_available else 0.0, 0.82, 0.92, 0.62, 0.65 if chemberta_available else 1.0),
        'role': 'SMILES semantic expert for cold-start drug generalization',
    }
    esm2_available = protein_embeddings is not None
    esm2_coverage = float(esm2_status.get('coverage', 0.0))
    sources['esm2_cached'] = {
        'available': bool(esm2_available),
        'coverage': esm2_coverage,
        'missing': int(esm2_status.get('missing', len(protein_ids))),
        'truncated': int(esm2_status.get('truncated', 0)),
        'supports_final_performance': 0.88,
        'continuity_potential': 0.95,
        'interpretability': 0.70,
        'computational_cost': 0.10 if esm2_available else 1.0,
        'expert_score': expert_source_score(esm2_coverage, 0.88, 0.95, 0.70, 0.10 if esm2_available else 1.0),
        'role': 'protein language and contact-informed semantic expert',
    }
    return {
        'dataset': dataset,
        'n_drugs': len(drug_ids),
        'n_proteins': len(protein_ids),
        'sources': sources,
        'recommendation': {
            'ready_for_long_training': bool(esm2_available),
            'highest_priority_missing_source': None if chemberta_available else 'chemberta2_cached',
            'long_training_policy': 'run after esm2_cached is available; include chemberta2_cached when local model or precomputed drug embeddings are available',
        },
    }


def save_audit(output_dir, audit):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'biochemical_prior_audit.json'), 'w', encoding='utf-8') as handle:
        json.dump(audit, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Build aligned ChemBERTa-2 and ESM2 cache files for HEAL-DTI.')
    parser.add_argument('--dataset', type=str, default='Davis', choices=['Davis', 'KIBA', 'DrugBank'])
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--data_seed', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--drug_embedding_map', type=str, default='')
    parser.add_argument('--protein_embedding_map', type=str, default='')
    parser.add_argument('--chemberta_model', type=str, default='')
    parser.add_argument('--esm_model', type=str, default='')
    parser.add_argument('--esm2_cache_dir', type=str, default='')
    parser.add_argument('--local_files_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--drug_max_length', type=int, default=256)
    parser.add_argument('--protein_max_length', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    interactions, _, _ = load_interactions(args.dataset, args.dataset_dir, args.data_seed)
    drug_ids, smiles, protein_ids, sequences = ordered_entities(interactions)
    output_dir = args.output_dir or os.path.join('data', args.dataset.lower())
    esm2_dir = args.esm2_cache_dir or os.path.join('data', args.dataset.lower(), 'esm2')

    drug_embeddings = align_embedding_map(drug_ids, load_embedding_map(args.drug_embedding_map))
    protein_embeddings = align_embedding_map(protein_ids, load_embedding_map(args.protein_embedding_map))
    esm2_status = {'available': protein_embeddings is not None, 'coverage': 1.0 if protein_embeddings is not None else 0.0, 'missing': 0 if protein_embeddings is not None else len(protein_ids), 'truncated': 0}
    if protein_embeddings is None:
        protein_embeddings, esm2_status = load_esm2_pooled_cache(protein_ids, esm2_dir)

    if drug_embeddings is None and args.chemberta_model:
        drug_embeddings = encode_texts_with_transformers(smiles, args.chemberta_model, args.batch_size, args.drug_max_length, args.device, args.local_files_only)
    if protein_embeddings is None and args.esm_model:
        protein_embeddings = encode_texts_with_transformers(sequences, args.esm_model, args.batch_size, args.protein_max_length, args.device, args.local_files_only)
        esm2_status = {'available': True, 'coverage': 1.0, 'missing': 0, 'truncated': 0}

    if drug_embeddings is None and protein_embeddings is None:
        raise ValueError('No embeddings were provided or generated. Set --drug_embedding_map/--protein_embedding_map or --chemberta_model/--esm_model.')
    if drug_embeddings is not None and drug_embeddings.size(0) != len(drug_ids):
        raise ValueError(f'Drug embedding count mismatch: {drug_embeddings.size(0)} vs {len(drug_ids)}')
    if protein_embeddings is not None and protein_embeddings.size(0) != len(protein_ids):
        raise ValueError(f'Protein embedding count mismatch: {protein_embeddings.size(0)} vs {len(protein_ids)}')

    save_cache(output_dir, args.dataset, drug_embeddings, protein_embeddings)
    audit = build_audit(args.dataset, drug_ids, protein_ids, drug_embeddings, protein_embeddings, esm2_status)
    save_audit(output_dir, audit)
    print({'dataset': args.dataset, 'output_dir': output_dir, 'n_drugs': len(drug_ids), 'n_proteins': len(protein_ids), 'drug_cache': drug_embeddings is not None, 'protein_cache': protein_embeddings is not None})


if __name__ == '__main__':
    main()
