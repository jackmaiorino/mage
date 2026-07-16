from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "kernel" / "python" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import generate_pauper_manifests as manifests  # noqa: E402


EXPECTED_SPECS = (
    ("Wildfire", "Wildfire", "Deck - Jund Wildfire.dek", "cff35798ff724888a9e5a4520dd55e70b0c628a55908697aa116089d8fd980a5"),
    ("Rally", "Rally", "Deck - Mono Red Rally.dek", "4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834"),
    ("Affinity", "Affinity", "Deck - Grixis Affinity.dek", "4a41135ac6d14960e75ddce8e9980c0505c0b71a9c08a2e10578a10d2fcf8801"),
    ("Elves", "Elves", "Deck - Elves.dek", "6b040933c9b3506536e7dc71c94dcaf5f16c7ade43a3d0f7f9b240be6deb0d87"),
    ("Spy", "SpyCombo", "Deck - Spy Combo.dek", "f08177d5ed133b18312f59649d1155e15b5074ababeaabcdf3f31ded650308ba"),
    ("Burn", "Burn", "Deck - Mono-Red Burn.dek", "4ebba6b42bb27a0ea55001cee133aada81f0dffd8661b46b012fc5026675aa32"),
    ("Terror", "Terror", "Deck - Mono-Blue Terror.dek", "8ba22b67b843bc49a421e1c2814c4dd24a04ab2b45131ec7876a8312115a9fda"),
    ("CawGates", "CawGates", "Deck - Caw-Gates.dek", "72c2bbf76a7fd219349a0ad81c44dc6166b4a797a1f66fe9b5a5de79aa6cdc14"),
    ("Faeries", "Faeries", "Deck - Mono-Blue Faeries.dek", "8cb962c4ccee6a5f8c0c70fc27c17d13323d13606c82b9b12b8985aa87e0f344"),
)

MISSING_SPY_RECORDS = {
    "Balustrade Spy",
    "Dread Return",
    "Elves of Deep Shadow",
    "Faerie Macabre",
    "Flaring Pain",
    "Fume Spitter",
    "Gatecreeper Vine",
    "Healer of the Glade",
    "Land Grant",
    "Lotleth Giant",
    "Lotus Petal",
    "Mesmeric Fiend",
    "Overgrown Battlement",
    "Sagu Wildling",
    "Saruli Caretaker",
    "Tinder Wall",
    "Troll of Khazad-dum",
    "Wall of Roots",
}

MISSING_SPY_MAIN = {
    "Balustrade Spy": 4,
    "Dread Return": 2,
    "Elves of Deep Shadow": 2,
    "Gatecreeper Vine": 3,
    "Land Grant": 4,
    "Lotleth Giant": 2,
    "Lotus Petal": 2,
    "Mesmeric Fiend": 2,
    "Overgrown Battlement": 4,
    "Sagu Wildling": 4,
    "Saruli Caretaker": 4,
    "Tinder Wall": 2,
    "Troll of Khazad-dum": 1,
    "Wall of Roots": 3,
}


def load(path: Path):
    return manifests.loads_json_strict(path.read_bytes(), context=path.as_posix())


def roster_from_xml(path: Path) -> tuple[Counter[str], Counter[str]]:
    mainboard: Counter[str] = Counter()
    sideboard: Counter[str] = Counter()
    for row in ElementTree.parse(path).getroot().findall("Cards"):
        target = sideboard if row.attrib["Sideboard"] == "true" else mainboard
        target[row.attrib["Name"]] += int(row.attrib["Quantity"])
    return mainboard, sideboard


def roster_from_zone(zone: dict) -> Counter[str]:
    return Counter({row["name"]: row["count"] for row in zone["cards"]})


class PauperPoolManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pool_path = REPO_ROOT / manifests.POOL_PATH
        cls.support_path = REPO_ROOT / manifests.SUPPORT_PATH
        cls.registry_path = REPO_ROOT / manifests.REGISTRY_PATH
        cls.pool = load(cls.pool_path)
        cls.support = load(cls.support_path)
        cls.registry = load(cls.registry_path)

    def test_exact_java_order_keys_paths_hashes_and_protocol(self) -> None:
        self.assertEqual(
            tuple(
                (spec.deck_id, spec.source_key, spec.filename, spec.source_sha256)
                for spec in manifests.DECK_SPECS
            ),
            EXPECTED_SPECS,
        )
        self.assertEqual(self.pool["schema"], "kernel_pauper_pool/v1")
        self.assertEqual(self.pool["protocol"], "canonical-mainboard-bo1/v1")
        self.assertEqual(
            self.pool["source"],
            {
                "java_factory_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/DeterminizationSampler.java",
                "java_factory_method": "DeterminizationSampler.pauperDefaults",
                "java_factory_file_sha256": "0df59e3f934aaafc46835411e3fc53cf060a63cceb03c4921e52c35f4d55669d",
                "java_factory_method_sha256": "a5fc8d84f7fa70f1c41c9ce0f50e892cb4d68119313128f54e14316a01febd7b",
                "source_hash_normalization": "utf8_text_crlf_v1",
            },
        )
        self.assertEqual(
            self.pool["materialization"],
            {"order": "utf8_card_name_then_copy_ordinal", "copy_ordinal_base": 1},
        )
        actual = tuple(
            (
                deck["id"],
                deck["source_key"],
                Path(deck["source_path"]).name,
                deck["source_sha256"],
            )
            for deck in self.pool["decks"]
        )
        self.assertEqual(actual, EXPECTED_SPECS)
        self.assertEqual([deck["order"] for deck in self.pool["decks"]], list(range(1, 10)))
        for deck in self.pool["decks"]:
            self.assertNotIn("\\", deck["source_path"])
            source = REPO_ROOT / deck["source_path"]
            self.assertEqual(
                manifests.canonical_source_sha256(
                    source.read_bytes(), context=deck["source_path"]
                ),
                deck["source_sha256"],
            )
        # The generator validates this declaration rather than merely trusting
        # its own duplicate list of constants.
        manifests.build_pool_manifest(REPO_ROOT)

    def test_rosters_are_exact_aggregates_and_materialization_is_canonical(self) -> None:
        main_names: set[str] = set()
        side_names: set[str] = set()
        main_copies = 0
        side_copies = 0
        for deck in self.pool["decks"]:
            source_main, source_side = roster_from_xml(REPO_ROOT / deck["source_path"])
            for zone_name, source_counts, expected_copies in (
                ("mainboard", source_main, 60),
                ("sideboard", source_side, 15),
            ):
                zone = deck[zone_name]
                generated_counts = roster_from_zone(zone)
                self.assertEqual(generated_counts, source_counts, (deck["id"], zone_name))
                self.assertEqual(zone["copy_count"], expected_copies)
                self.assertEqual(sum(generated_counts.values()), expected_copies)
                self.assertEqual(zone["unique_card_count"], len(generated_counts))
                self.assertEqual(
                    [row["name"] for row in zone["cards"]],
                    sorted(generated_counts, key=lambda name: name.encode("utf-8")),
                )
                for row in zone["cards"]:
                    self.assertIsInstance(row["name"], str)
                    self.assertTrue(row["name"])
                    self.assertEqual(row["name"], row["name"].strip())
                    self.assertIs(type(row["count"]), int)
                    self.assertGreater(row["count"], 0)
                expected_materialization = [
                    {"name": name, "copy_ordinal": ordinal}
                    for name in sorted(generated_counts, key=lambda value: value.encode("utf-8"))
                    for ordinal in range(1, generated_counts[name] + 1)
                ]
                self.assertEqual(zone["materialized_cards"], expected_materialization)
            main_names.update(source_main)
            side_names.update(source_side)
            main_copies += sum(source_main.values())
            side_copies += sum(source_side.values())
        self.assertEqual((len(main_names), len(side_names), len(main_names | side_names)), (121, 36, 150))
        self.assertEqual((main_copies, side_copies), (540, 135))
        self.assertEqual(
            self.pool["totals"],
            {
                "deck_count": 9,
                "mainboard_unique_cards": 121,
                "sideboard_unique_cards": 36,
                "pool_unique_cards": 150,
                "mainboard_copies": 540,
                "sideboard_copies": 135,
            },
        )
        caw_source = roster_from_xml(
            REPO_ROOT / manifests.DECK_BASE_PATH / "Deck - Caw-Gates.dek"
        )[0]
        caw_manifest = next(deck for deck in self.pool["decks"] if deck["id"] == "CawGates")
        self.assertEqual(caw_source["Island"], 4)
        self.assertEqual(roster_from_zone(caw_manifest["mainboard"])["Island"], 4)
        self.assertEqual(
            sum(1 for row in caw_manifest["mainboard"]["cards"] if row["name"] == "Island"),
            1,
        )

    def test_json_loader_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaises(manifests.ManifestError):
            manifests.loads_json_strict(b'{"x":1,"x":2}', context="duplicate")
        with self.assertRaises(manifests.ManifestError):
            manifests.loads_json_strict(b'{"x":NaN}', context="nonfinite")
        with self.assertRaises(manifests.ManifestError):
            manifests.loads_json_strict(b'{"x":1e999}', context="overflowed-float")

    def test_source_hash_contract_is_line_ending_invariant_and_utf8_strict(self) -> None:
        source = REPO_ROOT / manifests.DECK_BASE_PATH / EXPECTED_SPECS[0][2]
        raw = source.read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        lf = text.encode("utf-8")
        crlf = text.replace("\n", "\r\n").encode("utf-8")
        bare_cr = text.replace("\n", "\r").encode("utf-8")
        expected = EXPECTED_SPECS[0][3]
        for candidate in (lf, crlf, bare_cr):
            self.assertEqual(
                manifests.canonical_source_sha256(candidate, context="line-ending-test"),
                expected,
            )
        self.assertNotEqual(
            manifests.canonical_source_sha256(b"complete", context="final-newline-test"),
            manifests.canonical_source_sha256(b"complete\n", context="final-newline-test"),
        )
        with self.assertRaises(manifests.ManifestError):
            manifests.canonical_source_bytes(b"\xff", context="bad-utf8")

    def test_java_factory_validation_rejects_commented_regex_bypass(self) -> None:
        source = (REPO_ROOT / manifests.JAVA_FACTORY_PATH).read_text(
            encoding="utf-8", errors="strict"
        )
        mutated = source.replace("        paths.put(\"", "        // paths.put(\"")
        self.assertEqual(mutated.count("        // paths.put(\""), 9)
        mutated = mutated.replace(
            "        return loadArchetypes(paths);",
            "        paths.put(new String(\"NotCanonical\"), base + \"/Deck - Nope.dek\");\n"
            "        return loadArchetypes(paths);",
            1,
        )

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / manifests.JAVA_FACTORY_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(mutated, encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(manifests.ManifestError, "source drifted"):
                manifests._validate_java_factory(Path(temporary))

    def test_java_factory_validation_rejects_block_commented_canonical_copy(self) -> None:
        source = (REPO_ROOT / manifests.JAVA_FACTORY_PATH).read_text(
            encoding="utf-8", errors="strict"
        ).replace("\r\n", "\n").replace("\r", "\n")
        signature = "    public static DeterminizationSampler pauperDefaults() {"
        start = source.index(signature)
        return_start = source.index("        return loadArchetypes(paths);", start)
        end = source.index("\n    }", return_start) + len("\n    }")
        canonical_method = source[start:end]
        changed_method = canonical_method.replace(
            'base + "/Deck - Jund Wildfire.dek"',
            'base + "/Deck - Nope.dek"',
            1,
        )
        mutated = source.replace(
            canonical_method,
            "    /*\n" + canonical_method + "\n    */\n" + changed_method,
            1,
        )

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / manifests.JAVA_FACTORY_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(mutated, encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(manifests.ManifestError, "source drifted"):
                manifests._validate_java_factory(Path(temporary))

    def test_registry_membership_exactly_matches_all_nine_rosters(self) -> None:
        self.assertEqual(self.registry["version"], manifests.REGISTRY_SCHEMA_VERSION)
        expected_pool_decks = [spec[2] for spec in EXPECTED_SPECS]
        self.assertEqual(self.registry["pool_decks"], expected_pool_decks)
        expected_membership: dict[str, list[str]] = {}
        pool_names: set[str] = set()
        for deck in self.pool["decks"]:
            names = {
                row["name"]
                for zone in (deck["mainboard"], deck["sideboard"])
                for row in zone["cards"]
            }
            pool_names.update(names)
            for name in names:
                expected_membership.setdefault(name, []).append(Path(deck["source_path"]).name)
        registry_cards = {row["name"]: row for row in self.registry["cards"]}
        non_tokens = {name for name, row in registry_cards.items() if not row.get("is_token", False)}
        self.assertEqual(non_tokens, pool_names - MISSING_SPY_RECORDS)
        self.assertEqual(
            self.registry["unresolved"],
            sorted(MISSING_SPY_RECORDS, key=lambda name: name.encode("utf-8")),
        )
        for name in non_tokens:
            self.assertEqual(registry_cards[name]["decks"], expected_membership[name], name)
        for token_name in ("Blood Token", "Human Soldier Token", "Samurai Token"):
            self.assertTrue(registry_cards[token_name]["is_token"])
            self.assertEqual(registry_cards[token_name]["decks"], [])
        spy_declared = {
            name
            for name, row in registry_cards.items()
            if "Deck - Spy Combo.dek" in row.get("decks", [])
        }
        self.assertEqual(
            spy_declared,
            {
                "Forest",
                "Generous Ent",
                "Lead the Stampede",
                "Masked Vandal",
                "Quirion Ranger",
                "Swamp",
                "Vitu-Ghazi Inspector",
                "Winding Way",
            },
        )

    def test_missing_spy_records_are_explicit_and_exact(self) -> None:
        missing_rows = {
            row["name"]: row
            for row in self.support["cards"]
            if row["registry_status"] == "missing"
        }
        self.assertEqual(set(missing_rows), MISSING_SPY_RECORDS)
        self.assertEqual(list(missing_rows), self.registry["unresolved"])
        spy = next(deck for deck in self.pool["decks"] if deck["id"] == "Spy")
        spy_main = roster_from_zone(spy["mainboard"])
        missing_main = {name: spy_main[name] for name in MISSING_SPY_MAIN}
        self.assertEqual(missing_main, MISSING_SPY_MAIN)
        self.assertEqual(len(missing_main), 14)
        self.assertEqual(sum(missing_main.values()), 39)
        for row in missing_rows.values():
            self.assertEqual(row["declared_decks"], [])
            self.assertEqual(row["expected_decks"], ["Deck - Spy Combo.dek"])
            self.assertFalse(row["registry_membership_matches"])
            self.assertEqual(row["support_status"], "no_effect")
            self.assertEqual(row["blockers"], ["missing_registry_record", "no_effect_program"])

    def test_support_is_complete_hashed_and_pins_copy_totals(self) -> None:
        self.assertEqual(self.support["schema"], "kernel_pauper_support/v1")
        self.assertEqual(self.support["protocol"], "canonical-mainboard-bo1/v1")
        pool_raw = self.pool_path.read_bytes()
        registry_raw = self.registry_path.read_bytes()
        self.assertEqual(
            self.support["inputs"],
            {
                "pool_path": "kernel/data/pauper_pool_v1.json",
                "pool_raw_sha256": hashlib.sha256(pool_raw).hexdigest(),
                "registry_path": "kernel/data/cards_v1.json",
                "registry_raw_sha256": hashlib.sha256(registry_raw).hexdigest(),
            },
        )
        pool_names = {
            row["name"]
            for deck in self.pool["decks"]
            for zone in (deck["mainboard"], deck["sideboard"])
            for row in zone["cards"]
        }
        support_names = [row["name"] for row in self.support["cards"]]
        self.assertEqual(support_names, sorted(pool_names, key=lambda name: name.encode("utf-8")))
        self.assertEqual(len(support_names), 150)
        self.assertEqual(
            self.support["totals"],
            {
                "pool_cards": 150,
                "full_cards": 33,
                "partial_cards": 0,
                "no_effect_cards": 117,
                "token_dependencies": 3,
            },
        )
        expected_copy_totals = [
            {"deck_id": "Wildfire", "full": 7, "partial": 0, "no_effect": 53, "total": 60},
            {"deck_id": "Rally", "full": 60, "partial": 0, "no_effect": 0, "total": 60},
            {"deck_id": "Affinity", "full": 8, "partial": 0, "no_effect": 52, "total": 60},
            {"deck_id": "Elves", "full": 17, "partial": 0, "no_effect": 43, "total": 60},
            {"deck_id": "Spy", "full": 8, "partial": 0, "no_effect": 52, "total": 60},
            {"deck_id": "Burn", "full": 60, "partial": 0, "no_effect": 0, "total": 60},
            {"deck_id": "Terror", "full": 22, "partial": 0, "no_effect": 38, "total": 60},
            {"deck_id": "CawGates", "full": 8, "partial": 0, "no_effect": 52, "total": 60},
            {"deck_id": "Faeries", "full": 24, "partial": 0, "no_effect": 36, "total": 60},
        ]
        self.assertEqual(self.support["deck_mainboard_copy_totals"], expected_copy_totals)
        self.assertEqual(
            next(
                row
                for row in self.support["deck_mainboard_copy_totals"]
                if row["deck_id"] == "Rally"
            ),
            {"deck_id": "Rally", "full": 60, "partial": 0, "no_effect": 0, "total": 60},
        )
        registry_cards = {row["name"]: row for row in self.registry["cards"]}
        for row in self.support["cards"]:
            name = row["name"]
            if name in registry_cards:
                self.assertEqual(row["registry_status"], "present")
                self.assertEqual(row["declared_decks"], registry_cards[name]["decks"])
                self.assertEqual(row["expected_decks"], registry_cards[name]["decks"])
                self.assertTrue(row["registry_membership_matches"])
            if row["support_status"] == "full":
                self.assertEqual(row["blockers"], [])
            else:
                self.assertTrue(row["blockers"])
        chain = next(row for row in self.support["cards"] if row["name"] == "Chain Lightning")
        self.assertEqual(chain["support_status"], "full")
        self.assertEqual(chain["blockers"], [])
        self.assertEqual(
            self.support["token_dependencies"],
            [
                {
                    "name": "Blood Token",
                    "required_by": ["Voldaren Epicure"],
                    "registry_status": "present",
                    "expected_decks": [],
                    "declared_decks": [],
                    "registry_membership_matches": True,
                    "support_status": "full",
                    "blockers": [],
                },
                {
                    "name": "Human Soldier Token",
                    "required_by": ["Rally at the Hornburg"],
                    "registry_status": "present",
                    "expected_decks": [],
                    "declared_decks": [],
                    "registry_membership_matches": True,
                    "support_status": "full",
                    "blockers": [],
                },
                {
                    "name": "Samurai Token",
                    "required_by": ["Experimental Synthesizer"],
                    "registry_status": "present",
                    "expected_decks": [],
                    "declared_decks": [],
                    "registry_membership_matches": True,
                    "support_status": "full",
                    "blockers": [],
                },
            ],
        )

    def test_registry_capability_corruption_and_token_demotion_fail_closed(self) -> None:
        _pool, rosters = manifests.build_pool_manifest(REPO_ROOT)

        invalid = json.loads(json.dumps(self.registry))
        next(row for row in invalid["cards"] if row["name"] == "Island")[
            "engine_capability"
        ] = "future_magic"
        with self.assertRaisesRegex(manifests.ManifestError, "invalid engine_capability"):
            manifests.normalize_registry(invalid, rosters)

        demoted_token = json.loads(json.dumps(self.registry))
        next(row for row in demoted_token["cards"] if row["name"] == "Blood Token")[
            "engine_capability"
        ] = "no_effect"
        with self.assertRaisesRegex(
            manifests.ManifestError, "Blood Token.*full engine_capability"
        ):
            manifests.normalize_registry(demoted_token, rosters)

        self.assertEqual(
            manifests._support_status(
                "Counterspell", registry_card={"engine_capability": "partial"}
            ),
            ("partial", ["partial_program"]),
        )
        self.assertEqual(
            manifests._support_status("Counterspell", registry_card={}),
            ("no_effect", ["no_effect_program"]),
        )

    def test_generated_bytes_are_exact_and_corruption_fails_closed(self) -> None:
        expected = manifests.expected_outputs(REPO_ROOT)
        for relative_path, expected_bytes in expected.items():
            self.assertEqual((REPO_ROOT / relative_path).read_bytes(), expected_bytes)
        manifests.check_outputs(REPO_ROOT)

        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            required = [
                manifests.JAVA_FACTORY_PATH,
                manifests.REGISTRY_PATH,
                manifests.POOL_PATH,
                manifests.SUPPORT_PATH,
                *(Path(spec.source_path) for spec in manifests.DECK_SPECS),
            ]
            for relative_path in required:
                target = temporary_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO_ROOT / relative_path, target)
            manifests.check_outputs(temporary_root)
            for relative_path in (
                manifests.REGISTRY_PATH,
                manifests.POOL_PATH,
                manifests.SUPPORT_PATH,
            ):
                with self.subTest(relative_path=relative_path.as_posix()):
                    path = temporary_root / relative_path
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.assertRaises(manifests.ManifestError):
                        manifests.check_outputs(temporary_root)
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
