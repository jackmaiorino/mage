"""
generate_card_embeddings.py

Generates 32-dim PCA-compressed text embeddings for a set of MTG cards using
OpenAI text-embedding-3-large, then saves them as card_embeddings.json in the
profile's models directory.

Usage:
    python generate_card_embeddings.py --profile Pauper-Elves \
        --decklist ../../decks/PauperSubset/decklist.txt

    python generate_card_embeddings.py --profile Vintage-Cube \
        --cube ../../decks/Vintage/Cube/MTGOVintageCube.dck

Requirements:
    pip install openai scikit-learn requests

Environment:
    OPENAI_API_KEY - required
    MODEL_PROFILE  - fallback profile name if --profile not given

Scryfall bulk data is downloaded automatically on first run and cached at
MLPythonCode/scryfall_oracle_cards.json (auto-refreshed if >7 days old).
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.resolve()
_RL_BASE = _SCRIPT_DIR.parent  # …/mage/player/ai/rl
_SCRYFALL_CACHE = _SCRIPT_DIR / "scryfall_oracle_cards.json"
_SCRYFALL_BULK_URL = "https://api.scryfall.com/bulk-data/oracle-cards"
_EMBED_DIM = 32            # final PCA output dimension
_OPENAI_MODEL = "text-embedding-3-large"
_CACHE_MAX_AGE_DAYS = 7


def profile_models_dir(profile: str) -> Path:
    env_override = os.getenv("RL_MODELS_DIR", "").strip()
    if env_override:
        return Path(env_override)
    return _RL_BASE / "profiles" / profile / "models"


# ---------------------------------------------------------------------------
# Scryfall helpers
# ---------------------------------------------------------------------------

def download_scryfall_bulk() -> None:
    """Download the oracle-cards bulk JSON from Scryfall and cache it."""
    import requests
    print("Fetching Scryfall bulk-data index…")
    resp = requests.get(_SCRYFALL_BULK_URL, timeout=30)
    resp.raise_for_status()
    download_url = resp.json()["download_uri"]
    print(f"Downloading oracle cards from {download_url} (this may take a minute)…")
    with requests.get(download_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(_SCRYFALL_CACHE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:.0f}%", end="\r", flush=True)
    print(f"\nSaved to {_SCRYFALL_CACHE}")


def ensure_scryfall_cache() -> None:
    if _SCRYFALL_CACHE.exists():
        age_days = (time.time() - _SCRYFALL_CACHE.stat().st_mtime) / 86400
        if age_days < _CACHE_MAX_AGE_DAYS:
            return
        print(f"Scryfall cache is {age_days:.0f} days old, refreshing…")
    download_scryfall_bulk()


def build_scryfall_lookup() -> dict[str, dict]:
    """Return {card_name: card_data} from the cached bulk file."""
    ensure_scryfall_cache()
    print("Loading Scryfall card data…")
    with open(_SCRYFALL_CACHE, encoding="utf-8") as f:
        cards = json.load(f)
    lookup: dict[str, dict] = {}
    canonical_count = 0
    for card in cards:
        name = card.get("name", "")
        if name:
            canonical_count += 1
            lookup[name] = card
            lookup[normalise_card_name(name)] = card
            # Split-card / MDFC fallback aliases (e.g. "The Modern Age // Vector Glider")
            if " // " in name:
                for part in name.split(" // "):
                    part = part.strip()
                    if part:
                        lookup[part] = card
                        lookup[normalise_card_name(part)] = card
    print(f"  {canonical_count} cards loaded ({len(lookup)} lookup keys incl. aliases).")
    return lookup


def normalise_card_name(name: str) -> str:
    """Normalise for fuzzy matching against Scryfall names."""
    value = unicodedata.normalize("NFKD", name)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.strip().lower()


def card_text(card_data: dict) -> str:
    """Compose the text string we embed for a card."""
    parts = []
    if card_data.get("name"):
        parts.append(card_data["name"])
    if card_data.get("type_line"):
        parts.append(card_data["type_line"])
    if card_data.get("oracle_text"):
        parts.append(card_data["oracle_text"])
    elif card_data.get("card_faces"):
        face_texts = [f.get("oracle_text", "") for f in card_data["card_faces"]]
        parts.append(" // ".join(t for t in face_texts if t))
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Card name extraction from deck / cube files
# ---------------------------------------------------------------------------

def card_names_from_decklist(path: Path) -> list[str]:
    """
    Read a decklist.txt that contains one .dek path per line,
    then extract all card names from those .dek files.
    """
    names: set[str] = set()
    base = path.parent
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Preferred format: one deck file path per line.
            dek_path = base / line
            try:
                if dek_path.exists():
                    names.update(card_names_from_dek(dek_path.resolve()))
                    continue
            except OSError:
                # Fall through to inline-card parsing below.
                pass

            # Tolerate card-list inputs (e.g. "1 Lightning Bolt") when users
            # pass a cube list to --decklist by mistake.
            inline_name = parse_card_line(line)
            if inline_name:
                names.add(inline_name)
            else:
                print(f"  Warning: deck file not found: {dek_path.resolve()}", file=sys.stderr)
    return sorted(names)


def parse_card_line(line: str) -> str:
    """
    Parse a line that may contain a counted card entry and return the card name.
    Supported examples:
      1 Lightning Bolt
      SB: 2 [SET:123] Card Name
    """
    pattern = re.compile(r"^(?:SB:\s*)?\d+\s+\[.*?\]\s+(.+)$")
    simple = re.compile(r"^(?:SB:\s*)?\d+\s+(.+)$")
    match = pattern.match(line) or simple.match(line)
    return match.group(1).strip() if match else ""


def card_names_from_dek(path: Path) -> list[str]:
    """
    Parse an XMage .dek file. Lines look like:
      4 [SET:NNN] Card Name
    or (sideboard marker):
      SB: 2 [SET:NNN] Card Name
    """
    names: set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  Warning: deck file not found: {path}", file=sys.stderr)
        return []
    except OSError as exc:
        print(f"  Warning: could not read deck file {path}: {exc}", file=sys.stderr)
        return []

    # XMage deck files are commonly XML with <Cards ... Name="..."/> entries.
    stripped = content.lstrip()
    if stripped.startswith("<"):
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(content)
            for card_elem in root.findall(".//Cards"):
                name = (card_elem.attrib.get("Name") or "").strip()
                if name:
                    names.add(name)
            return list(names)
        except Exception:
            # Fall back to text-line parsing below for non-XML variants.
            pass

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = parse_card_line(line)
        if name:
            names.add(name)
    return list(names)


def card_names_from_cube(path: Path) -> list[str]:
    """
    Parse an XMage cube .dck file. Same format as .dek but may also have
    lines like:  NAME:Vintage Cube  or  1 [SET:NNN] Card Name
    """
    return card_names_from_dek(path)


# ---------------------------------------------------------------------------
# OpenAI embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    """Call OpenAI embeddings API in batches, return list of raw vectors."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing Python package 'openai'. Install with: pip install openai") from exc
    client = OpenAI(api_key=api_key)

    all_embeddings: list[list[float]] = []
    batch_size = 100  # text-embedding-3-large supports up to 2048 inputs/request
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        print(f"  Embedding batch {start // batch_size + 1} / {(len(texts) - 1) // batch_size + 1}"
              f"  ({start + 1}–{min(start + len(batch), len(texts))} of {len(texts)})…")
        try:
            response = client.embeddings.create(model=_OPENAI_MODEL, input=batch)
        except Exception as exc:
            # Keep this broad so we can provide a clear, dependency-light error path
            # across openai package versions.
            status_code = getattr(exc, "status_code", None)
            err_code = None
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    err_code = err.get("code")
            if status_code == 429 and err_code == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI API quota exceeded (insufficient_quota). "
                    "Check API key/project billing and usage limits at https://platform.openai.com."
                ) from exc
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
    return all_embeddings


# ---------------------------------------------------------------------------
# PCA compression
# ---------------------------------------------------------------------------

def pca_compress(embeddings: list[list[float]], n_components: int) -> list[list[float]]:
    """Fit PCA on the provided embeddings and compress to n_components dims."""
    import numpy as np
    from sklearn.decomposition import PCA

    X = np.array(embeddings, dtype="float32")
    effective_components = min(n_components, X.shape[0], X.shape[1])
    if effective_components < n_components:
        print(f"  Warning: only {X.shape[0]} cards — using {effective_components} PCA components")

    pca = PCA(n_components=effective_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {effective_components} components, {variance:.1%} variance retained")

    # Pad to _EMBED_DIM if we had fewer samples than components
    if effective_components < n_components:
        pad = np.zeros((X_reduced.shape[0], n_components - effective_components), dtype="float32")
        X_reduced = np.concatenate([X_reduced, pad], axis=1)

    # L2-normalise rows so embeddings are unit vectors
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_reduced = X_reduced / norms

    return X_reduced.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate card text embeddings for RL training")
    parser.add_argument("--profile", default=os.getenv("MODEL_PROFILE", ""),
                        help="Profile name (e.g. Pauper-Elves, Vintage-Cube)")
    parser.add_argument("--decklist", default="",
                        help="Path to decklist.txt (lists .dek files, one per line)")
    parser.add_argument("--cube", default="",
                        help="Path to cube .dck file")
    parser.add_argument("--dek", default="",
                        help="Path to a single .dek file")
    parser.add_argument("--output", default="",
                        help="Override output path for card_embeddings.json")
    parser.add_argument("--embed-dim", type=int, default=_EMBED_DIM,
                        help=f"PCA output dimension (default: {_EMBED_DIM})")
    args = parser.parse_args()

    if not args.profile:
        print("Error: --profile is required (or set MODEL_PROFILE env var)", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY env var is not set", file=sys.stderr)
        sys.exit(1)

    # Resolve card names
    card_names: list[str] = []
    if args.decklist:
        card_names = card_names_from_decklist(Path(args.decklist))
    elif args.cube:
        card_names = card_names_from_cube(Path(args.cube))
    elif args.dek:
        card_names = card_names_from_dek(Path(args.dek))
    else:
        print("Error: one of --decklist, --cube, or --dek is required", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(card_names)} unique card names")
    if not card_names:
        print(
            "Error: no card names found. Check that --decklist/--dek paths are valid and deck files are parseable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve Scryfall oracle text
    scryfall = build_scryfall_lookup()
    texts: list[str] = []
    missing: list[str] = []
    resolved_names: list[str] = []
    for name in card_names:
        data = scryfall.get(name) or scryfall.get(normalise_card_name(name))
        if data is None:
            missing.append(name)
            texts.append(name)  # embed just the name as fallback
        else:
            texts.append(card_text(data))
        resolved_names.append(name)

    if missing:
        print(f"  Warning: {len(missing)} cards not found in Scryfall: {missing[:10]}"
              + (" …" if len(missing) > 10 else ""))

    # Embed
    print(f"\nEmbedding {len(texts)} cards with {_OPENAI_MODEL}…")
    try:
        raw_embeddings = embed_texts(texts, api_key)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # PCA compress
    print(f"\nCompressing to {args.embed_dim} dimensions via PCA…")
    compressed = pca_compress(raw_embeddings, args.embed_dim)

    # Build output map
    result: dict[str, list[float]] = {
        name: [round(float(x), 6) for x in vec]
        for name, vec in zip(resolved_names, compressed)
    }

    # Write output
    if args.output:
        out_path = Path(args.output)
    else:
        models_dir = profile_models_dir(args.profile)
        models_dir.mkdir(parents=True, exist_ok=True)
        out_path = models_dir / "card_embeddings.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(result)} card embeddings to {out_path}")


if __name__ == "__main__":
    main()
