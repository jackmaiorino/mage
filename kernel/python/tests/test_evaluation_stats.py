from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError, fields
from decimal import ROUND_DOWN, localcontext

import mtg_kernel_rl
from mtg_kernel_rl.determinism import derive_evaluation_bootstrap_seed
from mtg_kernel_rl.evaluation_stats import (
    BootstrapSummary,
    ScoreSummary,
    SignTestResult,
    WilsonInterval,
    _SplitMix64,
    _bootstrap_replicate_sums,
    _unbiased_index,
    bootstrap_pair_half_points,
    exact_two_sided_sign_test,
    score_pair_half_points,
    wilson_interval,
)


class EvaluationStatsTest(unittest.TestCase):
    def test_splitmix64_known_sequence(self) -> None:
        rng = _SplitMix64(0)
        self.assertEqual(
            [rng.next_u64() for _ in range(6)],
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
                0x1B39_896A_51A8_749B,
                0x53CB_9F0C_747E_A2EA,
            ],
        )

    def test_unbiased_index_rejects_incomplete_upper_bucket(self) -> None:
        population_size = 3
        limit = ((1 << 64) // population_size) * population_size
        values = iter((limit, 7))
        calls = 0

        def next_u64() -> int:
            nonlocal calls
            calls += 1
            return next(values)

        self.assertEqual(_unbiased_index(population_size, next_u64), 1)
        self.assertEqual(calls, 2)

    def test_bootstrap_golden_vector_and_digest(self) -> None:
        result = bootstrap_pair_half_points([0, 1, 3, 4], 0, 1_000)
        self.assertIsInstance(result, BootstrapSummary)
        self.assertEqual(result.pair_count, 4)
        self.assertEqual(result.total_half_points, 8)
        self.assertEqual(result.estimate, 0.5)
        self.assertEqual(result.confidence_level, 0.95)
        self.assertEqual(result.bootstrap_seed, 0)
        self.assertEqual(result.bootstrap_replicates, 1_000)
        self.assertEqual(
            _bootstrap_replicate_sums((0, 1, 3, 4), 0, 1_000)[:8],
            (8, 8, 11, 13, 4, 8, 5, 11),
        )
        self.assertEqual(
            result.replicate_sums_sha256,
            "056651411c69bbf75716ba8bf7ac87ad3b8ded34b7011537da0df2ccb7a3654e",
        )
        self.assertEqual(result.lower_percentile_index, 24)
        self.assertEqual(result.upper_percentile_index, 975)
        self.assertEqual(result.lower_sum, 2)
        self.assertEqual(result.upper_sum, 15)
        self.assertEqual(result.lower, 0.125)
        self.assertEqual(result.upper, 0.9375)
        with self.assertRaises(FrozenInstanceError):
            result.lower = 0.0  # type: ignore[misc]

    def test_bootstrap_n_one_is_degenerate(self) -> None:
        result = bootstrap_pair_half_points(
            [3],
            bootstrap_seed=0xFFFF_FFFF_FFFF_FFFF,
            bootstrap_replicates=1_000,
        )
        self.assertEqual(_bootstrap_replicate_sums((3,), 0xFFFF_FFFF_FFFF_FFFF, 1_000), (3,) * 1_000)
        self.assertEqual(result.estimate, 0.75)
        self.assertEqual(result.lower_sum, 3)
        self.assertEqual(result.upper_sum, 3)
        self.assertEqual(result.lower, 0.75)
        self.assertEqual(result.upper, 0.75)

    def test_bootstrap_derived_seed_supplemental_oracle(self) -> None:
        seed = derive_evaluation_bootstrap_seed(71_501)
        result = bootstrap_pair_half_points([0, 1, 2, 3, 4], seed, 1_000)
        self.assertEqual(result.lower_sum, 4)
        self.assertEqual(result.upper_sum, 16)
        self.assertEqual(result.lower.hex(), "0x1.999999999999ap-3")
        self.assertEqual(result.upper.hex(), "0x1.999999999999ap-1")
        self.assertEqual(
            result.replicate_sums_sha256,
            "4c5dc351d474c2628a247e23861cf2959aa22775bf699aa5ee52b7114b7af9c9",
        )

    def test_bootstrap_ignores_and_preserves_global_random_state(self) -> None:
        random.seed(12_345)
        before = random.getstate()
        first = bootstrap_pair_half_points([0, 1, 3, 4], 99, 1_000)
        self.assertEqual(random.getstate(), before)
        for _ in range(100):
            random.random()
        second = bootstrap_pair_half_points([0, 1, 3, 4], 99, 1_000)
        self.assertEqual(first, second)

    def test_bootstrap_digest_preserves_observation_order(self) -> None:
        forward = bootstrap_pair_half_points([0, 1, 3, 4], 0, 1_000)
        reverse = bootstrap_pair_half_points([4, 3, 1, 0], 0, 1_000)
        self.assertNotEqual(forward.replicate_sums_sha256, reverse.replicate_sums_sha256)

    def test_exact_two_sided_sign_test_excludes_ties(self) -> None:
        result = exact_two_sided_sign_test([4, 4, 4, 0, 2])
        self.assertIsInstance(result, SignTestResult)
        self.assertEqual(result.wins, 3)
        self.assertEqual(result.losses, 1)
        self.assertEqual(result.ties, 1)
        self.assertEqual(result.non_ties, 4)
        self.assertEqual((result.p_value_numerator, result.p_value_denominator), (5, 8))
        self.assertEqual(result.p_value, 0.625)
        self.assertEqual(exact_two_sided_sign_test([4, 4, 4, 0, 0]).p_value, 1.0)
        all_ties = exact_two_sided_sign_test([2, 2, 2])
        self.assertEqual((all_ties.p_value_numerator, all_ties.p_value_denominator), (1, 1))
        self.assertEqual(all_ties.non_ties, 0)

    def test_sign_test_extreme_exact_fraction(self) -> None:
        result = exact_two_sided_sign_test([4, 4, 4])
        self.assertEqual((result.p_value_numerator, result.p_value_denominator), (1, 4))
        self.assertEqual(result.p_value, 0.25)

    def test_sign_test_retains_exact_tail_when_float_underflows(self) -> None:
        result = exact_two_sided_sign_test([4] * 50_000)
        self.assertEqual(result.p_value_numerator, 1)
        self.assertEqual(result.p_value_denominator.bit_length(), 50_000)
        self.assertEqual(result.p_value, 0.0)

    def test_wilson_interval_fixed_float_hex_goldens(self) -> None:
        goldens = {
            (0, 10): ("0x0.0p+0", "0x0.0p+0", "0x1.1c318eebe9e79p-2"),
            (5, 10): ("0x1.0000000000000p-1", "0x1.e48aeb11b030ap-3", "0x1.86dd453b93f3dp-1"),
            (10, 10): ("0x1.0000000000000p+0", "0x1.71e7388a0b0c3p-1", "0x1.0000000000000p+0"),
            (1, 2): ("0x1.0000000000000p-1", "0x1.83332751478d3p-4", "0x1.cf999b15d70e6p-1"),
        }
        for (successes, trials), expected in goldens.items():
            with self.subTest(successes=successes, trials=trials):
                result = wilson_interval(successes, trials)
                self.assertIsInstance(result, WilsonInterval)
                self.assertEqual((result.estimate_hex, result.lower_hex, result.upper_hex), expected)

    def test_wilson_n_one_and_hostile_ambient_decimal_context_independence(self) -> None:
        with localcontext() as context:
            context.prec = 6
            context.rounding = ROUND_DOWN
            context.Emin = -1
            context.Emax = 1
            for signal in context.traps:
                context.traps[signal] = True
            before = (
                context.prec,
                context.rounding,
                context.Emin,
                context.Emax,
                context.capitals,
                context.clamp,
                dict(context.flags),
                dict(context.traps),
            )
            zero = wilson_interval(0, 1)
            one = wilson_interval(1, 1)
            after = (
                context.prec,
                context.rounding,
                context.Emin,
                context.Emax,
                context.capitals,
                context.clamp,
                dict(context.flags),
                dict(context.traps),
            )
            self.assertEqual(after, before)
        self.assertEqual(
            (zero.estimate_hex, zero.lower_hex, zero.upper_hex),
            ("0x0.0p+0", "0x0.0p+0", "0x1.963f2b137a224p-1"),
        )
        self.assertEqual(
            (one.estimate_hex, one.lower_hex, one.upper_hex),
            ("0x1.0000000000000p+0", "0x1.a70353b21776fp-3", "0x1.0000000000000p+0"),
        )

    def test_same_primary_total_can_have_different_paired_distributions(self) -> None:
        variable = score_pair_half_points([0, 4], 7, 1_000)
        tied = score_pair_half_points([2, 2], 7, 1_000)
        self.assertIsInstance(variable, ScoreSummary)
        self.assertEqual(variable.total_half_points, tied.total_half_points)
        self.assertEqual(variable.estimate, tied.estimate)
        self.assertEqual(variable.estimate, 0.5)
        self.assertEqual((variable.sign_test.wins, variable.sign_test.losses), (1, 1))
        self.assertEqual((tied.sign_test.wins, tied.sign_test.losses), (0, 0))
        self.assertNotEqual(variable.bootstrap.replicate_sums_sha256, tied.bootstrap.replicate_sums_sha256)
        self.assertEqual((tied.bootstrap.lower, tied.bootstrap.upper), (0.5, 0.5))

    def test_pair_totals_cannot_identify_game_level_wilson_marginals(self) -> None:
        seat_sweeps = [(2, 0), (2, 0)]
        seat_draws = [(1, 1), (1, 1)]
        self.assertNotEqual(
            tuple(sum(game[seat] for game in seat_sweeps) for seat in (0, 1)),
            tuple(sum(game[seat] for game in seat_draws) for seat in (0, 1)),
        )
        sweep_totals = [sum(game) for game in seat_sweeps]
        draw_totals = [sum(game) for game in seat_draws]
        self.assertEqual(sweep_totals, draw_totals)
        self.assertEqual(
            score_pair_half_points(sweep_totals, 7, 1_000),
            score_pair_half_points(draw_totals, 7, 1_000),
        )
        self.assertNotIn("wilson", {field.name for field in fields(ScoreSummary)})

    def test_integrated_score_oracle_and_single_generator_materialization(self) -> None:
        pair_values = [4] * 10 + [2] * 3 + [0] * 2
        result = score_pair_half_points(
            (value for value in pair_values),
            derive_evaluation_bootstrap_seed(71_501),
            1_000,
        )
        self.assertEqual((result.pair_count, result.total_half_points), (15, 46))
        self.assertEqual(result.estimate.hex(), "0x1.8888888888889p-1")
        self.assertEqual((result.sign_test.wins, result.sign_test.losses, result.sign_test.ties), (10, 2, 3))
        self.assertEqual((result.sign_test.p_value_numerator, result.sign_test.p_value_denominator), (79, 2_048))
        self.assertEqual(result.sign_test.p_value.hex(), "0x1.3c00000000000p-5")
        self.assertEqual(
            (result.bootstrap.lower_percentile_index, result.bootstrap.upper_percentile_index),
            (24, 975),
        )
        self.assertEqual((result.bootstrap.lower_sum, result.bootstrap.upper_sum), (34, 56))
        self.assertEqual(result.bootstrap.lower.hex(), "0x1.2222222222222p-1")
        self.assertEqual(result.bootstrap.upper.hex(), "0x1.ddddddddddddep-1")
        self.assertEqual(
            result.bootstrap.replicate_sums_sha256,
            "6249ec465c9412cba617614d9508fdd37c931814b68d2b75cbf71f4d87121b1b",
        )

    def test_pair_and_bootstrap_bounds_reject_bool(self) -> None:
        bad_pairs = ([], [-1], [5], [True], [1.0])
        for pair_values in bad_pairs:
            with self.subTest(pair_values=pair_values), self.assertRaises((TypeError, ValueError)):
                bootstrap_pair_half_points(pair_values, 0, 1_000)  # type: ignore[arg-type]
            with self.subTest(sign_pairs=pair_values), self.assertRaises((TypeError, ValueError)):
                exact_two_sided_sign_test(pair_values)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            exact_two_sided_sign_test([2] * 50_001)
        for bad_seed in (True, -1, 2**64):
            with self.subTest(seed=bad_seed), self.assertRaises((TypeError, ValueError)):
                bootstrap_pair_half_points([2], bad_seed, 1_000)  # type: ignore[arg-type]
        for bad_replicates in (True, 999, 100_001):
            with self.subTest(replicates=bad_replicates), self.assertRaises((TypeError, ValueError)):
                bootstrap_pair_half_points([2], 0, bad_replicates)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            bootstrap_pair_half_points([2] * 501, 0, 100_000)
        for unordered in ({0, 1, 3, 4}, frozenset((0, 1, 3, 4)), {0: 4, 1: 0}):
            with self.subTest(unordered=type(unordered).__name__), self.assertRaises(TypeError):
                bootstrap_pair_half_points(unordered, 0, 1_000)  # type: ignore[arg-type]
            with self.subTest(sign_unordered=type(unordered).__name__), self.assertRaises(TypeError):
                exact_two_sided_sign_test(unordered)  # type: ignore[arg-type]

    def test_wilson_bounds_reject_bool(self) -> None:
        for args in ((True, 1), (0, True), (-1, 1), (2, 1), (0, 0), (0, 50_001)):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                wilson_interval(*args)  # type: ignore[arg-type]

    def test_package_root_exports_only_stable_evaluation_api(self) -> None:
        expected = {
            "BootstrapSummary",
            "ScoreSummary",
            "SignTestResult",
            "WilsonInterval",
            "bootstrap_pair_half_points",
            "exact_two_sided_sign_test",
            "score_pair_half_points",
            "wilson_interval",
        }
        self.assertTrue(expected.issubset(set(mtg_kernel_rl.__all__)))
        self.assertFalse(hasattr(mtg_kernel_rl, "_SplitMix64"))
        self.assertFalse(hasattr(mtg_kernel_rl, "_unbiased_index"))


if __name__ == "__main__":
    unittest.main()
