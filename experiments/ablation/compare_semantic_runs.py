import argparse
import json

import torch


def load_row(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not payload or payload[0].get("status") != "ok":
        raise RuntimeError(f"run is not successful: {path}")
    return payload[0]


def semantic_hashes(row):
    probe = row.get("semantic_output_probe") or {}
    return [
        (item.get("index"), item.get("shape"), item.get("sha256"))
        for item in probe.get("rows", [])
    ]


def semantic_names(row):
    probe = row.get("semantic_output_probe") or {}
    return [item.get("name") for item in probe.get("rows", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()

    reference = load_row(args.reference)
    candidate = load_row(args.candidate)
    reference_logits = torch.load(reference["logits_path"], map_location="cpu")
    candidate_logits = torch.load(candidate["logits_path"], map_location="cpu")
    logits_equal = torch.equal(reference_logits, candidate_logits)
    max_abs = (
        (reference_logits.float() - candidate_logits.float()).abs().max().item()
        if reference_logits.shape == candidate_logits.shape
        else float("inf")
    )
    reference_hashes = semantic_hashes(reference)
    candidate_hashes = semantic_hashes(candidate)
    hashes_equal = reference_hashes == candidate_hashes
    names_equal = semantic_names(reference) == semantic_names(candidate)
    result = {
        "reference": args.reference,
        "candidate": args.candidate,
        "logits_equal": logits_equal,
        "logits_max_abs": max_abs,
        "reference_layer_outputs": len(reference_hashes),
        "candidate_layer_outputs": len(candidate_hashes),
        "layer_hashes_equal": hashes_equal,
        "layer_names_equal": names_equal,
    }
    print(json.dumps(result, indent=2))
    if not logits_equal or not hashes_equal:
        raise AssertionError("semantic runs differ")


if __name__ == "__main__":
    main()
