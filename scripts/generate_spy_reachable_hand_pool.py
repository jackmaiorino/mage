#!/usr/bin/env python3
"""Generate randomized Spy Combo opening-hand starts for line-search imitation."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, List


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DECK = (
    REPO
    / "Mage.Server.Plugins"
    / "Mage.Player.AIRL"
    / "src"
    / "mage"
    / "player"
    / "ai"
    / "decks"
    / "Pauper"
    / "Deck - Spy Combo.dek"
)

SEED_HANDS = [
    "Forest,Forest,Swamp,Balustrade Spy,Tinder Wall,Elves of Deep Shadow,Wall of Roots",
    "Forest,Forest,Swamp,Balustrade Spy,Tinder Wall,Saruli Caretaker,Quirion Ranger",
    "Swamp,Land Grant,Winding Way,Sagu Wildling,Balustrade Spy,Gatecreeper Vine,Overgrown Battlement",
    "Land Grant,Winding Way,Generous Ent,Balustrade Spy,Wall of Roots,Elves of Deep Shadow,Saruli Caretaker",
    "Lead the Stampede,Winding Way,Generous Ent,Generous Ent,Elves of Deep Shadow,Saruli Caretaker,Tinder Wall",
    "Dread Return,Winding Way,Generous Ent,Sagu Wildling,Gatecreeper Vine,Masked Vandal,Overgrown Battlement",
    "Lead the Stampede,Winding Way,Generous Ent,Masked Vandal,Mesmeric Fiend,Wall of Roots,Quirion Ranger",
    "Swamp,Lead the Stampede,Winding Way,Lotus Petal,Lotleth Giant,Overgrown Battlement,Overgrown Battlement",
]

LAND_LIKE = {"Forest", "Swamp", "Land Grant", "Generous Ent", "Troll of Khazad-dum"}
TRUE_LANDS = {"Forest", "Swamp"}
ACCEL = {
    "Lotus Petal",
    "Tinder Wall",
    "Elves of Deep Shadow",
    "Saruli Caretaker",
    "Overgrown Battlement",
    "Wall of Roots",
}
CREATURES = {
    "Mesmeric Fiend",
    "Overgrown Battlement",
    "Saruli Caretaker",
    "Gatecreeper Vine",
    "Sagu Wildling",
    "Generous Ent",
    "Balustrade Spy",
    "Lotleth Giant",
    "Wall of Roots",
    "Masked Vandal",
    "Quirion Ranger",
    "Troll of Khazad-dum",
    "Tinder Wall",
    "Elves of Deep Shadow",
}
DIG = {"Winding Way", "Lead the Stampede"}


def parse_deck(path: Path) -> List[str]:
    root = ET.parse(path).getroot()
    cards: List[str] = []
    for node in root.iter():
        if not node.tag.endswith("Cards"):
            continue
        if node.attrib.get("Sideboard", "false").lower() == "true":
            continue
        name = node.attrib.get("Name", "").strip()
        qty = int(node.attrib.get("Quantity", "0"))
        cards.extend([name] * qty)
    if len(cards) < 60:
        raise RuntimeError(f"Parsed only {len(cards)} main-deck cards from {path}")
    return cards


def hand_from_line(line: str) -> List[str]:
    return [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]


def legal_hand(hand: Iterable[str], deck_counts: Counter[str]) -> bool:
    counts = Counter(hand)
    return len(counts) > 0 and all(count <= deck_counts[name] for name, count in counts.items())


def reachable_score(hand: List[str]) -> int:
    c = Counter(hand)
    score = 0
    score += 4 * c["Balustrade Spy"]
    score += 2 * sum(c[name] for name in DIG)
    score += 2 * sum(c[name] for name in TRUE_LANDS)
    score += sum(c[name] for name in LAND_LIKE)
    score += sum(c[name] for name in ACCEL)
    score += sum(c[name] for name in CREATURES)
    if c["Swamp"] or c["Lotus Petal"] or c["Elves of Deep Shadow"]:
        score += 3
    if c["Dread Return"] and c["Lotleth Giant"]:
        score += 1
    return score


def looks_reachable(hand: List[str], strict: bool) -> bool:
    c = Counter(hand)
    has_spy_or_dig = c["Balustrade Spy"] > 0 or sum(c[name] for name in DIG) > 0
    if not has_spy_or_dig:
        return False
    if sum(c[name] for name in LAND_LIKE) < 1:
        return False
    if sum(c[name] for name in CREATURES) < 2:
        return False
    if strict and reachable_score(hand) < 12:
        return False
    return True


def mutate_seed(seed: List[str], deck: List[str], deck_counts: Counter[str], rng: random.Random) -> List[str]:
    hand = list(seed)
    replacements = rng.choice([0, 0, 1, 1, 1, 2])
    for _ in range(replacements):
        idx = rng.randrange(len(hand))
        for _attempt in range(50):
            candidate = rng.choice(deck)
            trial = list(hand)
            trial[idx] = candidate
            if legal_hand(trial, deck_counts):
                hand = trial
                break
    return hand


def generate(deck: List[str], count: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    deck_counts = Counter(deck)
    out: List[List[str]] = []
    seen = set()

    def add(hand: List[str]) -> bool:
        if len(hand) != 7 or not legal_hand(hand, deck_counts) or not looks_reachable(hand, strict=True):
            return False
        key = tuple(sorted(hand))
        if key in seen:
            return False
        seen.add(key)
        out.append(hand)
        return True

    seeds = [hand_from_line(line) for line in SEED_HANDS]
    for hand in seeds:
        add(hand)
    while len(out) < min(count, 512):
        seed_hand = rng.choice(seeds)
        add(mutate_seed(seed_hand, deck, deck_counts, rng))

    attempts = 0
    while len(out) < count and attempts < count * 1000:
        attempts += 1
        hand = rng.sample(deck, 7)
        if looks_reachable(hand, strict=True):
            add(hand)
    attempts = 0
    while len(out) < count and attempts < count * 1000:
        attempts += 1
        hand = rng.sample(deck, 7)
        if looks_reachable(hand, strict=False):
            add(hand)
    if len(out) < count:
        raise RuntimeError(f"Only generated {len(out)} hands; requested {count}")
    return out[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=6101)
    args = parser.parse_args()

    deck_path = args.deck if args.deck.is_absolute() else (REPO / args.deck)
    out_path = args.out if args.out.is_absolute() else (REPO / args.out)
    hands = generate(parse_deck(deck_path), max(1, args.count), args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(",".join(hand) for hand in hands) + "\n", encoding="utf-8")
    print(f"wrote {len(hands)} hands to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
