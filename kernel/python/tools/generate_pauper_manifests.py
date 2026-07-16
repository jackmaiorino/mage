#!/usr/bin/env python3
"""Generate and verify the canonical nine-deck Pauper pool manifests.

This tool is intentionally stdlib-only.  It treats the Java
``DeterminizationSampler.pauperDefaults()`` declaration and its nine checked-in
XMage deck files as the roster source of truth. Engine support is read from
each registry record's fail-closed ``engine_capability`` field, shared with the
Rust card-definition code generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree


POOL_SCHEMA = "kernel_pauper_pool/v1"
SUPPORT_SCHEMA = "kernel_pauper_support/v1"
REGISTRY_SCHEMA_VERSION = 2
PROTOCOL = "canonical-mainboard-bo1/v1"
SOURCE_HASH_NORMALIZATION = "utf8_text_crlf_v1"
MATERIALIZATION_ORDER = "utf8_card_name_then_copy_ordinal"
JAVA_FACTORY_FILE_SHA256 = "0df59e3f934aaafc46835411e3fc53cf060a63cceb03c4921e52c35f4d55669d"
JAVA_FACTORY_METHOD_SHA256 = "a5fc8d84f7fa70f1c41c9ce0f50e892cb4d68119313128f54e14316a01febd7b"

JAVA_FACTORY_PATH = Path(
    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/DeterminizationSampler.java"
)
DECK_BASE_PATH = Path(
    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper"
)
REGISTRY_PATH = Path("kernel/data/cards_v1.json")
POOL_PATH = Path("kernel/data/pauper_pool_v1.json")
SUPPORT_PATH = Path("kernel/data/pauper_support_v1.json")


class ManifestError(RuntimeError):
    """Raised when a source or generated manifest fails closed."""


@dataclass(frozen=True)
class DeckSpec:
    deck_id: str
    source_key: str
    filename: str
    source_sha256: str

    @property
    def source_path(self) -> str:
        return (DECK_BASE_PATH / self.filename).as_posix()


# This order is the protocol order and must match pauperDefaults() exactly.
DECK_SPECS = (
    DeckSpec("Wildfire", "Wildfire", "Deck - Jund Wildfire.dek", "cff35798ff724888a9e5a4520dd55e70b0c628a55908697aa116089d8fd980a5"),
    DeckSpec("Rally", "Rally", "Deck - Mono Red Rally.dek", "4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834"),
    DeckSpec("Affinity", "Affinity", "Deck - Grixis Affinity.dek", "4a41135ac6d14960e75ddce8e9980c0505c0b71a9c08a2e10578a10d2fcf8801"),
    DeckSpec("Elves", "Elves", "Deck - Elves.dek", "6b040933c9b3506536e7dc71c94dcaf5f16c7ade43a3d0f7f9b240be6deb0d87"),
    # The science-facing id is Spy; the historical Java source key is SpyCombo.
    DeckSpec("Spy", "SpyCombo", "Deck - Spy Combo.dek", "f08177d5ed133b18312f59649d1155e15b5074ababeaabcdf3f31ded650308ba"),
    DeckSpec("Burn", "Burn", "Deck - Mono-Red Burn.dek", "4ebba6b42bb27a0ea55001cee133aada81f0dffd8661b46b012fc5026675aa32"),
    DeckSpec("Terror", "Terror", "Deck - Mono-Blue Terror.dek", "8ba22b67b843bc49a421e1c2814c4dd24a04ab2b45131ec7876a8312115a9fda"),
    DeckSpec("CawGates", "CawGates", "Deck - Caw-Gates.dek", "72c2bbf76a7fd219349a0ad81c44dc6166b4a797a1f66fe9b5a5de79aa6cdc14"),
    DeckSpec("Faeries", "Faeries", "Deck - Mono-Blue Faeries.dek", "8cb962c4ccee6a5f8c0c70fc27c17d13323d13606c82b9b12b8985aa87e0f344"),
)


TOKEN_DEPENDENCIES = (
    ("Blood Token", ("Voldaren Epicure",)),
    ("Human Soldier Token", ("Rally at the Hornburg",)),
    ("Samurai Token", ("Experimental Synthesizer",)),
)

EXPECTED_MAINBOARD_SUPPORT = {
    "Wildfire": {"full": 7, "partial": 0, "no_effect": 53},
    "Rally": {"full": 60, "partial": 0, "no_effect": 0},
    "Affinity": {"full": 8, "partial": 0, "no_effect": 52},
    "Elves": {"full": 13, "partial": 0, "no_effect": 47},
    "Spy": {"full": 4, "partial": 0, "no_effect": 56},
    "Burn": {"full": 60, "partial": 0, "no_effect": 0},
    "Terror": {"full": 16, "partial": 0, "no_effect": 44},
    "CawGates": {"full": 4, "partial": 0, "no_effect": 56},
    "Faeries": {"full": 18, "partial": 0, "no_effect": 42},
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_source_bytes(raw: bytes, *, context: str) -> bytes:
    """Return the portable full-text bytes used by the pinned deck hashes.

    Input must be strict UTF-8.  CRLF, LF, and bare-CR line boundaries are
    normalized to LF, then every LF is encoded as CRLF.  No final newline is
    added or removed, so the hash still covers the complete XML text.
    """

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManifestError(f"{context}: source is not strict UTF-8") from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\r\n").encode("utf-8")


def canonical_source_sha256(raw: bytes, *, context: str) -> str:
    return sha256_hex(canonical_source_bytes(raw, context=context))


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def loads_json_strict(raw: bytes | str, *, context: str) -> Any:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManifestError(f"{context}: JSON is not strict UTF-8") from exc
    else:
        text = raw

    def reject_constant(value: str) -> None:
        raise ManifestError(f"{context}: non-finite JSON constant {value!r}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ManifestError(f"{context}: non-finite JSON number {value!r}")
        return parsed

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"{context}: invalid JSON: {exc}") from exc


def dump_json(payload: Any) -> bytes:
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"cannot serialize manifest: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _utf8_sort(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _validate_java_factory(repo_root: Path) -> None:
    path = repo_root / JAVA_FACTORY_PATH
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read Java factory {path}: {exc}") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    file_sha256 = sha256_hex(text.encode("utf-8"))
    if file_sha256 != JAVA_FACTORY_FILE_SHA256:
        raise ManifestError(
            "Java factory source drifted: "
            f"expected {JAVA_FACTORY_FILE_SHA256}, got {file_sha256}"
        )
    signature = "    public static DeterminizationSampler pauperDefaults() {"
    return_line = "        return loadArchetypes(paths);"
    start = text.find(signature)
    if start < 0:
        raise ManifestError("pauperDefaults() declaration not found")
    return_start = text.find(return_line, start)
    if return_start < 0:
        raise ManifestError("pauperDefaults() return not found")
    closing_start = text.find("\n    }", return_start + len(return_line))
    if closing_start < 0:
        raise ManifestError("pauperDefaults() closing brace not found")
    method = text[start : closing_start + len("\n    }")]
    method_sha256 = sha256_hex(method.encode("utf-8"))
    if method_sha256 != JAVA_FACTORY_METHOD_SHA256:
        raise ManifestError(
            "pauperDefaults() method body drifted: "
            f"expected {JAVA_FACTORY_METHOD_SHA256}, got {method_sha256}"
        )
    body = method
    base_match = re.search(r'String base = "([^"]+)";', body)
    if base_match is None or base_match.group(1) != DECK_BASE_PATH.as_posix():
        raise ManifestError("pauperDefaults() deck base path drifted")
    actual = re.findall(r'paths\.put\("([^"]+)", base \+ "/([^"]+)"\);', body)
    expected = [(spec.source_key, spec.filename) for spec in DECK_SPECS]
    if actual != expected:
        raise ManifestError(f"pauperDefaults() order/path drift: expected {expected!r}, got {actual!r}")


def _parse_deck(repo_root: Path, spec: DeckSpec) -> tuple[Counter[str], Counter[str]]:
    path = repo_root / Path(spec.source_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read deck source {path}: {exc}") from exc
    actual_hash = canonical_source_sha256(raw, context=spec.source_path)
    if actual_hash != spec.source_sha256:
        raise ManifestError(
            f"{spec.source_path}: canonical source SHA-256 drift: "
            f"expected {spec.source_sha256}, got {actual_hash}"
        )
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ManifestError(f"{spec.source_path}: invalid deck XML: {exc}") from exc
    if root.tag != "Deck":
        raise ManifestError(f"{spec.source_path}: expected Deck root, got {root.tag!r}")

    mainboard: Counter[str] = Counter()
    sideboard: Counter[str] = Counter()
    for row_number, row in enumerate(root.findall("Cards"), start=1):
        name = row.attrib.get("Name")
        quantity_text = row.attrib.get("Quantity")
        sideboard_text = row.attrib.get("Sideboard")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ManifestError(f"{spec.source_path}: Cards row {row_number} has invalid Name")
        try:
            quantity = int(quantity_text or "")
        except ValueError as exc:
            raise ManifestError(
                f"{spec.source_path}: Cards row {row_number} has invalid Quantity"
            ) from exc
        if quantity <= 0 or str(quantity) != quantity_text:
            raise ManifestError(
                f"{spec.source_path}: Cards row {row_number} Quantity must be a positive integer"
            )
        if sideboard_text == "false":
            mainboard[name] += quantity
        elif sideboard_text == "true":
            sideboard[name] += quantity
        else:
            raise ManifestError(
                f"{spec.source_path}: Cards row {row_number} Sideboard must be true or false"
            )
    if sum(mainboard.values()) != 60 or sum(sideboard.values()) != 15:
        raise ManifestError(
            f"{spec.source_path}: expected 60+15 cards, got "
            f"{sum(mainboard.values())}+{sum(sideboard.values())}"
        )
    return mainboard, sideboard


def _zone_payload(counts: Counter[str]) -> dict[str, Any]:
    names = _utf8_sort(counts)
    cards = [{"name": name, "count": counts[name]} for name in names]
    materialized = [
        {"name": name, "copy_ordinal": ordinal}
        for name in names
        for ordinal in range(1, counts[name] + 1)
    ]
    return {
        "copy_count": sum(counts.values()),
        "unique_card_count": len(counts),
        "cards": cards,
        "materialized_cards": materialized,
    }


def build_pool_manifest(repo_root: Path) -> tuple[dict[str, Any], dict[str, tuple[Counter[str], Counter[str]]]]:
    _validate_java_factory(repo_root)
    rosters: dict[str, tuple[Counter[str], Counter[str]]] = {}
    decks: list[dict[str, Any]] = []
    all_main: set[str] = set()
    all_side: set[str] = set()
    main_copies = 0
    side_copies = 0
    for order, spec in enumerate(DECK_SPECS, start=1):
        mainboard, sideboard = _parse_deck(repo_root, spec)
        rosters[spec.deck_id] = (mainboard, sideboard)
        all_main.update(mainboard)
        all_side.update(sideboard)
        main_copies += sum(mainboard.values())
        side_copies += sum(sideboard.values())
        decks.append(
            {
                "order": order,
                "id": spec.deck_id,
                "source_key": spec.source_key,
                "source_path": spec.source_path,
                "source_sha256": spec.source_sha256,
                "mainboard": _zone_payload(mainboard),
                "sideboard": _zone_payload(sideboard),
            }
        )
    all_cards = all_main | all_side
    expected_totals = (9, 121, 36, 150, 540, 135)
    actual_totals = (
        len(decks),
        len(all_main),
        len(all_side),
        len(all_cards),
        main_copies,
        side_copies,
    )
    if actual_totals != expected_totals:
        raise ManifestError(f"canonical pool totals drift: expected {expected_totals}, got {actual_totals}")
    manifest = {
        "schema": POOL_SCHEMA,
        "protocol": PROTOCOL,
        "source": {
            "java_factory_path": JAVA_FACTORY_PATH.as_posix(),
            "java_factory_method": "DeterminizationSampler.pauperDefaults",
            "java_factory_file_sha256": JAVA_FACTORY_FILE_SHA256,
            "java_factory_method_sha256": JAVA_FACTORY_METHOD_SHA256,
            "source_hash_normalization": SOURCE_HASH_NORMALIZATION,
        },
        "materialization": {
            "order": MATERIALIZATION_ORDER,
            "copy_ordinal_base": 1,
        },
        "totals": {
            "deck_count": len(decks),
            "mainboard_unique_cards": len(all_main),
            "sideboard_unique_cards": len(all_side),
            "pool_unique_cards": len(all_cards),
            "mainboard_copies": main_copies,
            "sideboard_copies": side_copies,
        },
        "decks": decks,
    }
    return manifest, rosters


def _expected_memberships(
    rosters: dict[str, tuple[Counter[str], Counter[str]]]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for spec in DECK_SPECS:
        mainboard, sideboard = rosters[spec.deck_id]
        for name in mainboard.keys() | sideboard.keys():
            result.setdefault(name, []).append(spec.filename)
    return result


def normalize_registry(
    registry: dict[str, Any], rosters: dict[str, tuple[Counter[str], Counter[str]]]
) -> dict[str, Any]:
    if registry.get("version") != REGISTRY_SCHEMA_VERSION:
        raise ManifestError(
            f"cards_v1.json must have version {REGISTRY_SCHEMA_VERSION}"
        )
    cards = registry.get("cards")
    if not isinstance(cards, list):
        raise ManifestError("cards_v1.json cards must be a list")
    expected_memberships = _expected_memberships(rosters)
    seen: set[str] = set()
    registered_non_tokens: set[str] = set()
    non_token_count = 0
    token_names: set[str] = set()
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            raise ManifestError(f"cards_v1.json card {index} must be an object")
        name = card.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ManifestError(f"cards_v1.json card {index} has invalid name")
        if name in seen:
            raise ManifestError(f"cards_v1.json duplicate card name {name!r}")
        seen.add(name)
        capability = _registry_engine_capability(
            card, context=f"cards_v1.json card {name!r}"
        )
        if card.get("is_token") is True:
            token_names.add(name)
            if card.get("decks") != []:
                raise ManifestError(f"token {name!r} must have empty deck membership")
            if capability != "full":
                raise ManifestError(
                    f"token {name!r} must have full engine_capability"
                )
            continue
        non_token_count += 1
        registered_non_tokens.add(name)
        if name not in expected_memberships:
            raise ManifestError(f"registry-only non-token card {name!r} is outside the pinned pool")
        card["decks"] = list(expected_memberships[name])
    expected_token_names = {name for name, _producers in TOKEN_DEPENDENCIES}
    if non_token_count != 132 or token_names != expected_token_names:
        raise ManifestError(
            f"registry baseline drift: expected 132 deck cards and tokens "
            f"{sorted(expected_token_names)!r}, got {non_token_count} and {sorted(token_names)!r}"
        )
    registry["pool_decks"] = [spec.filename for spec in DECK_SPECS]
    registry["unresolved"] = _utf8_sort(set(expected_memberships) - registered_non_tokens)
    return registry


def _registry_engine_capability(card: dict[str, Any], *, context: str) -> str:
    capability = card.get("engine_capability", "no_effect")
    if capability not in {"no_effect", "partial", "full"}:
        raise ManifestError(
            f"{context} has invalid engine_capability {capability!r}; "
            "expected no_effect, partial, or full"
        )
    return capability


def _support_status(
    name: str, *, registry_card: dict[str, Any] | None
) -> tuple[str, list[str]]:
    if registry_card is None:
        return "no_effect", ["missing_registry_record", "no_effect_program"]
    status = _registry_engine_capability(registry_card, context=f"registry card {name!r}")
    if status == "full":
        return "full", []
    if status == "partial":
        return "partial", ["partial_program"]
    return "no_effect", ["no_effect_program"]


def build_support_manifest(
    *,
    rosters: dict[str, tuple[Counter[str], Counter[str]]],
    registry: dict[str, Any],
    pool_bytes: bytes,
    registry_bytes: bytes,
) -> dict[str, Any]:
    expected_memberships = _expected_memberships(rosters)
    pool_names = set(expected_memberships)
    registry_cards = {card["name"]: card for card in registry["cards"]}
    cards: list[dict[str, Any]] = []
    status_unique_counts = Counter()
    for name in _utf8_sort(pool_names):
        registry_card = registry_cards.get(name)
        registry_present = registry_card is not None
        status, blockers = _support_status(name, registry_card=registry_card)
        status_unique_counts[status] += 1
        mainboard = []
        sideboard = []
        for spec in DECK_SPECS:
            main_counts, side_counts = rosters[spec.deck_id]
            if name in main_counts:
                mainboard.append({"deck_id": spec.deck_id, "copies": main_counts[name]})
            if name in side_counts:
                sideboard.append({"deck_id": spec.deck_id, "copies": side_counts[name]})
        declared_decks = list(registry_card.get("decks", [])) if registry_present else []
        expected_decks = list(expected_memberships[name])
        cards.append(
            {
                "name": name,
                "mainboard": mainboard,
                "sideboard": sideboard,
                "registry_status": "present" if registry_present else "missing",
                "expected_decks": expected_decks,
                "declared_decks": declared_decks,
                "registry_membership_matches": registry_present and declared_decks == expected_decks,
                "support_status": status,
                "blockers": blockers,
            }
        )

    deck_mainboard_copy_totals: list[dict[str, Any]] = []
    for spec in DECK_SPECS:
        counts = Counter({"full": 0, "partial": 0, "no_effect": 0})
        mainboard, _sideboard = rosters[spec.deck_id]
        for name, copies in mainboard.items():
            status, _blockers = _support_status(name, registry_card=registry_cards.get(name))
            counts[status] += copies
        actual = {key: counts[key] for key in ("full", "partial", "no_effect")}
        if actual != EXPECTED_MAINBOARD_SUPPORT[spec.deck_id]:
            raise ManifestError(
                f"{spec.deck_id} mainboard support drift: "
                f"expected {EXPECTED_MAINBOARD_SUPPORT[spec.deck_id]!r}, got {actual!r}"
            )
        deck_mainboard_copy_totals.append(
            {"deck_id": spec.deck_id, **actual, "total": sum(actual.values())}
        )

    token_dependencies: list[dict[str, Any]] = []
    for token_name, producers in TOKEN_DEPENDENCIES:
        token = registry_cards.get(token_name)
        if token is None or token.get("is_token") is not True:
            raise ManifestError(f"required token {token_name!r} is absent or not marked as a token")
        if any(producer not in registry_cards for producer in producers):
            raise ManifestError(f"required token {token_name!r} has an absent producer")
        status, blockers = _support_status(token_name, registry_card=token)
        if status != "full":
            raise ManifestError(
                f"required token {token_name!r} must have full engine capability, got {status!r}"
            )
        token_dependencies.append(
            {
                "name": token_name,
                "required_by": list(producers),
                "registry_status": "present",
                "expected_decks": [],
                "declared_decks": list(token.get("decks", [])),
                "registry_membership_matches": token.get("decks") == [],
                "support_status": status,
                "blockers": blockers,
            }
        )

    return {
        "schema": SUPPORT_SCHEMA,
        "protocol": PROTOCOL,
        "inputs": {
            "pool_path": POOL_PATH.as_posix(),
            "pool_raw_sha256": sha256_hex(pool_bytes),
            "registry_path": REGISTRY_PATH.as_posix(),
            "registry_raw_sha256": sha256_hex(registry_bytes),
        },
        "totals": {
            "pool_cards": len(cards),
            "full_cards": status_unique_counts["full"],
            "partial_cards": status_unique_counts["partial"],
            "no_effect_cards": status_unique_counts["no_effect"],
            "token_dependencies": len(token_dependencies),
        },
        "deck_mainboard_copy_totals": deck_mainboard_copy_totals,
        "token_dependencies": token_dependencies,
        "cards": cards,
    }


def expected_outputs(repo_root: Path) -> dict[Path, bytes]:
    pool, rosters = build_pool_manifest(repo_root)
    registry_path = repo_root / REGISTRY_PATH
    try:
        registry_raw = registry_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read registry {registry_path}: {exc}") from exc
    registry = loads_json_strict(registry_raw, context=REGISTRY_PATH.as_posix())
    if not isinstance(registry, dict):
        raise ManifestError("cards_v1.json root must be an object")
    normalized_registry = normalize_registry(registry, rosters)
    registry_bytes = dump_json(normalized_registry)
    pool_bytes = dump_json(pool)
    support = build_support_manifest(
        rosters=rosters,
        registry=normalized_registry,
        pool_bytes=pool_bytes,
        registry_bytes=registry_bytes,
    )
    return {
        REGISTRY_PATH: registry_bytes,
        POOL_PATH: pool_bytes,
        SUPPORT_PATH: dump_json(support),
    }


def check_outputs(repo_root: Path) -> None:
    for relative_path, expected in expected_outputs(repo_root).items():
        path = repo_root / relative_path
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise ManifestError(f"cannot read generated file {path}: {exc}") from exc
        # Parse all JSON through the duplicate-key rejecting loader even when a
        # byte mismatch would already fail, so ambiguous JSON is never admitted.
        loads_json_strict(actual, context=relative_path.as_posix())
        if actual != expected:
            raise ManifestError(
                f"{relative_path.as_posix()} is stale or non-canonical; "
                "run generate_pauper_manifests.py --write"
            )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_outputs(repo_root: Path) -> None:
    outputs = expected_outputs(repo_root)
    for relative_path in (REGISTRY_PATH, POOL_PATH, SUPPORT_PATH):
        path = repo_root / relative_path
        expected = outputs[relative_path]
        if path.exists() and path.read_bytes() == expected:
            continue
        _atomic_write(path, expected)
    check_outputs(repo_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite generated files canonically")
    mode.add_argument("--check", action="store_true", help="fail unless generated files are current")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(),
        help="repository root (default: inferred from this script)",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.write:
            write_outputs(repo_root)
        else:
            check_outputs(repo_root)
    except ManifestError as exc:
        print(f"PAUPER_MANIFEST: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"PAUPER_MANIFEST: {'WROTE' if args.write else 'PASS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
