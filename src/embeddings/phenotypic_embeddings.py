from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd


# Provisional placeholder symptom descriptions until MDS-UPDRS access is approved.
# Replace these with official MDS-UPDRS symptom text once access is granted.
DEFAULT_MDS_UPDRS = [
    "Resting tremor affecting one or more limbs, mild and intermittently present.",
    "Rigidity with moderate resistance throughout passive movement of the limbs.",
    "Bradykinesia with reduced speed and amplitude during repetitive voluntary movement.",
    "Postural instability with impaired balance and gait during daily activities.",
    "Freezing of gait episodes causing hesitation during walking initiation or turning.",
    "Speech difficulties including hypophonia and reduced articulation clarity.",
]


DEFAULT_MODEL_NAME = "m42-health/Llama3-Med42-8B"
FALLBACK_MODEL_NAME = "meta-llama/Llama-3.2-1B"


class PhenotypeEmbeddingError(RuntimeError):
    pass



def _load_descriptions(input_path=None, text_col="description"):
    if input_path is None:
        return pd.DataFrame(
            {
                "phenotype_id": [f"P{i + 1}" for i in range(len(DEFAULT_MDS_UPDRS))],
                "description": DEFAULT_MDS_UPDRS,
                "data_source": "placeholder_until_mds_updrs_approval",
            }
        )

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Phenotype input file not found: {path}")

    if path.suffix.lower() == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(
            {
                "phenotype_id": [f"P{i + 1}" for i in range(len(lines))],
                "description": lines,
                "data_source": str(path),
            }
        )

    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {path}")

    df = df.copy()
    if "phenotype_id" not in df.columns:
        df["phenotype_id"] = [f"P{i + 1}" for i in range(len(df))]
    if "data_source" not in df.columns:
        df["data_source"] = str(path)

    return df[["phenotype_id", text_col, "data_source"]].rename(columns={text_col: "description"})



def _resolve_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer



def _load_llm_backbone(model_name, hf_token=None):
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "LLM phenotype embeddings require 'torch' and 'transformers'."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    tokenizer = _resolve_pad_token(tokenizer)

    model = None
    load_errors = []

    try:
        model = AutoModel.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            dtype=torch.float16 if torch.cuda.is_available() else None,
        )
    except Exception as exc:
        load_errors.append(f"AutoModel failed: {exc}")

    if model is None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                dtype=torch.float16 if torch.cuda.is_available() else None,
            )
        except Exception as exc:
            load_errors.append(f"AutoModelForCausalLM failed: {exc}")

    if model is None:
        joined = " | ".join(load_errors)
        raise PhenotypeEmbeddingError(
            f"Could not load model '{model_name}'. Details: {joined}"
        )

    if hasattr(model, "resize_token_embeddings"):
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device, torch



def _mean_pool(last_hidden_state, attention_mask, torch_module):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled



def llm_text_embeddings(
    texts,
    model_name=DEFAULT_MODEL_NAME,
    batch_size=4,
    max_length=512,
    hf_token=None,
    normalize=True,
):
    tokenizer, model, device, torch = _load_llm_backbone(model_name=model_name, hf_token=hf_token)

    all_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start : start + batch_size]]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(device) for k, v in toks.items()}

            outputs = model(**toks, output_hidden_states=False, return_dict=True)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                hidden = outputs.hidden_states[-1]
            elif hasattr(outputs, "logits") and outputs.logits is not None:
                # Fallback for some causal models_GNN if hidden states are not directly exposed.
                hidden = outputs.logits
                warnings.warn(
                    "Falling back to logits for phenotype embeddings because hidden states were unavailable. "
                    "This is less ideal than hidden-state pooling."
                )
            else:
                raise PhenotypeEmbeddingError(
                    f"Model '{model_name}' did not return usable hidden states for embedding extraction."
                )

            pooled = _mean_pool(hidden, toks["attention_mask"], torch)
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_batches.append(pooled.detach().cpu().numpy().astype(np.float32))

    return np.vstack(all_batches)



def generate_phenotypic_embeddings(
    output_csv,
    input_path=None,
    text_col="description",
    model_name=DEFAULT_MODEL_NAME,
    fallback_model_name=FALLBACK_MODEL_NAME,
    batch_size=4,
    max_length=512,
    hf_token=None,
    normalize=True,
):
    df = _load_descriptions(input_path=input_path, text_col=text_col)
    texts = df["description"].astype(str).tolist()

    used_model_name = model_name
    try:
        vectors = llm_text_embeddings(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            hf_token=hf_token,
            normalize=normalize,
        )
    except Exception as primary_exc:
        if fallback_model_name and fallback_model_name != model_name:
            warnings.warn(
                f"Primary phenotype model '{model_name}' failed ({primary_exc}). "
                f"Falling back to '{fallback_model_name}'."
            )
            vectors = llm_text_embeddings(
                texts=texts,
                model_name=fallback_model_name,
                batch_size=batch_size,
                max_length=max_length,
                hf_token=hf_token,
                normalize=normalize,
            )
            used_model_name = fallback_model_name
        else:
            raise

    emb_df = pd.DataFrame(vectors, columns=[f"pheno_emb_{i}" for i in range(vectors.shape[1])])
    out_df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    out_df["embedding_model"] = used_model_name
    out_df["embedding_pooling"] = "mean_pool_last_hidden_state"
    out_df["embedding_normalized"] = bool(normalize)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} phenotype embeddings to {out_path} using {used_model_name}")
    return out_df



def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate phenotype embeddings from symptom descriptions using MedLlama/Llama hidden-state pooling."
    )
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--text_col", type=str, default="description")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--fallback_model_name", type=str, default=FALLBACK_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=project_root / "data" / "processed" / "phenotype_embeddings.csv",
    )
    args = parser.parse_args()

    generate_phenotypic_embeddings(
        output_csv=args.output_csv,
        input_path=args.input_path,
        text_col=args.text_col,
        model_name=args.model_name,
        fallback_model_name=args.fallback_model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        hf_token=args.hf_token,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
