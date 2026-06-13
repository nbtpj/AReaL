import unittest

from augmentation_pipeline import (
    generate_question_variants,
    generate_question_variant,
    generate_synonym_variant,
    generate_typo_noise_variant,
    generate_word_shuffle_variant,
    validate_variant_constraints,
)


class AugmentationPipelineTest(unittest.TestCase):
    def test_fixed_seed_is_bit_identical(self) -> None:
        question = "Alice will solve the small task quickly and show the result."
        protected_tokens = ("Alice",)

        first = generate_question_variants(question, num_variants=4, seed=11, protected_tokens=protected_tokens)
        second = generate_question_variants(question, num_variants=4, seed=11, protected_tokens=protected_tokens)

        self.assertEqual(first, second)

    def test_returns_exact_requested_variant_count(self) -> None:
        question = "Solve the small puzzle and find the result."
        variants = generate_question_variants(question, num_variants=5, seed=7)
        self.assertEqual(len(variants), 5)

    def test_different_seeds_produce_different_variants(self) -> None:
        question = "Solve the small puzzle and find the large answer quickly."
        a = generate_question_variants(question, num_variants=3, seed=1)
        b = generate_question_variants(question, num_variants=3, seed=2)
        self.assertNotEqual(a, b)

    def test_negative_num_variants_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_variants must be positive"):
            generate_question_variants("Solve this.", num_variants=-1, seed=1)

    def test_rejects_when_no_eligible_synonym_token_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "no eligible token available"):
            generate_synonym_variant("123 456", seed=3)

    def test_protected_tokens_and_numbers_remain_unchanged(self) -> None:
        question = "Alice will solve 3 small tasks quickly."
        variants = generate_question_variants(question, num_variants=4, seed=5, protected_tokens=("Alice",))

        for variant in variants:
            self.assertIn("Alice", variant)
            self.assertIn("3", variant)
            validate_variant_constraints(question, variant, protected_tokens=("Alice",))

    def test_rejects_variant_that_changes_named_entity(self) -> None:
        source = "Alice will solve the task."
        invalid_variant = "Bob will solve the task."
        with self.assertRaisesRegex(ValueError, "protected token"):
            validate_variant_constraints(source, invalid_variant, protected_tokens=("Alice",))

    def test_rejects_variant_that_changes_numeric_constraint(self) -> None:
        source = "Alice solves 3 tasks."
        invalid_variant = "Alice solves 4 tasks."
        with self.assertRaisesRegex(ValueError, "numeric constraints"):
            validate_variant_constraints(source, invalid_variant, protected_tokens=("Alice",))

    def test_only_approved_mapping_tokens_are_replaced(self) -> None:
        question = "Compute a theorem proof."
        variant = generate_synonym_variant(
            question,
            seed=9,
            synonym_map={"compute": "calculate"},
            protected_tokens=(),
        )
        self.assertEqual(variant, "Calculate a theorem proof.")

    def test_word_shuffle_preserves_protected_tokens_and_numbers(self) -> None:
        question = "Alice will solve 3 small tasks quickly and show the result."
        variant = generate_word_shuffle_variant(question, seed=4, protected_tokens=("Alice",))

        self.assertNotEqual(variant, question)
        self.assertIn("Alice", variant)
        self.assertIn("3", variant)
        validate_variant_constraints(question, variant, protected_tokens=("Alice",))

    def test_typo_noise_preserves_protected_tokens_and_numbers(self) -> None:
        question = "Alice will solve 3 small tasks quickly and show the result."
        variant = generate_typo_noise_variant(question, seed=4, protected_tokens=("Alice",))

        self.assertNotEqual(variant, question)
        self.assertIn("Alice", variant)
        self.assertIn("3", variant)
        validate_variant_constraints(question, variant, protected_tokens=("Alice",))

    def test_generate_question_variant_supports_three_modes(self) -> None:
        question = "Alice will solve 3 small tasks quickly and show the large result."

        variants = [
            generate_question_variant(
                question,
                seed=4,
                operator=operator,
                protected_tokens=("Alice",),
            )
            for operator in ("synonym_substitution", "word_shuffle", "typo_noise")
        ]

        self.assertEqual(len(variants), 3)
        self.assertEqual(len(set(variants)), 3)

    def test_generate_question_variants_cycles_configured_modes(self) -> None:
        question = "Alice will solve 3 small tasks quickly and show the large result."

        variants = generate_question_variants(
            question,
            num_variants=3,
            seed=4,
            protected_tokens=("Alice",),
            operators=("synonym_substitution", "word_shuffle", "typo_noise"),
        )

        self.assertEqual(len(variants), 3)
        self.assertEqual(len(set(variants)), 3)


if __name__ == "__main__":
    unittest.main()
