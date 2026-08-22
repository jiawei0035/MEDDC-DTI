"""Convert a repeated-row legacy DTI table into normalized TSV files."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    drugs = {}
    proteins = {}
    pairs = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Malformed row {line_number}: expected at least five columns")
            drug_id, protein_id, smiles, sequence, label = fields[-5:]
            if drug_id in drugs and drugs[drug_id] != smiles:
                raise ValueError(f"Conflicting SMILES for {drug_id}")
            if protein_id in proteins and proteins[protein_id] != sequence:
                raise ValueError(f"Conflicting sequence for {protein_id}")
            drugs[drug_id] = smiles
            proteins[protein_id] = sequence
            pairs.append((drug_id, protein_id, int(label)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "drugs.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("drug_id\tsmiles\n")
        for drug_id in sorted(drugs):
            handle.write(f"{drug_id}\t{drugs[drug_id]}\n")
    with (args.output_dir / "proteins.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("protein_id\tsequence\n")
        for protein_id in sorted(proteins):
            handle.write(f"{protein_id}\t{proteins[protein_id]}\n")
    with (args.output_dir / "interactions.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("drug_id\tprotein_id\tlabel\n")
        for drug_id, protein_id, label in pairs:
            handle.write(f"{drug_id}\t{protein_id}\t{label}\n")
    print(f"Wrote {len(drugs)} drugs, {len(proteins)} proteins, and {len(pairs)} interactions")


if __name__ == "__main__":
    main()
