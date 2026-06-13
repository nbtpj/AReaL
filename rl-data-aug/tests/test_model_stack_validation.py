import unittest

from augmentation_pipeline import generate_question_variants
from model_stack_validation import (
    Qwen3InitializationSpec,
    validate_prompt_tokenization,
    validate_qwen3_initialization_spec,
    validate_qwen3_stack_and_tokenization,
)
from training_config import TrainingStackConfig


class _NonDeterministicTokenizer:
    def __init__(self) -> None:
        self._counter = 0

    def encode(self, prompt: str) -> list[int]:
        self._counter += 1
        return [len(prompt), self._counter]


class ModelStackValidationTest(unittest.TestCase):
    def test_accepts_valid_stack_and_augmented_prompt_tokenization(self) -> None:
        config = TrainingStackConfig(
            augmentation_enabled=True,
            augmentation_operators=("synonym_substitution",),
            num_variants=3,
        )
        spec = Qwen3InitializationSpec(
            policy_model_path="models/Qwen3-1.7B/policy",
            reference_model_path="models/Qwen3-1.7B/reference",
            tokenizer_path="models/Qwen3-1.7B/tokenizer",
        )
        prompts = generate_question_variants(
            question="Alice will solve 3 small tasks quickly.",
            num_variants=3,
            seed=9,
            protected_tokens=("Alice",),
        )

        validate_qwen3_stack_and_tokenization(
            config=config,
            spec=spec,
            prompts=prompts,
            encode=lambda text: [ord(char) for char in text],
            require_existing_paths=False,
        )

    def test_rejects_non_qwen3_policy_path(self) -> None:
        spec = Qwen3InitializationSpec(
            policy_model_path="models/llama-3-8b/policy",
            reference_model_path="models/Qwen3-1.7B/reference",
            tokenizer_path="models/Qwen3-1.7B/tokenizer",
        )
        with self.assertRaisesRegex(ValueError, "policy_model_path must resolve to Qwen3-1.7B"):
            validate_qwen3_initialization_spec(spec=spec)

    def test_rejects_non_qwen3_reference_path(self) -> None:
        spec = Qwen3InitializationSpec(
            policy_model_path="models/Qwen3-1.7B/policy",
            reference_model_path="models/Qwen3-8B/reference",
            tokenizer_path="models/Qwen3-1.7B/tokenizer",
        )
        with self.assertRaisesRegex(ValueError, "reference_model_path must resolve to Qwen3-1.7B"):
            validate_qwen3_initialization_spec(spec=spec)

    def test_rejects_missing_required_init_path_when_required(self) -> None:
        spec = Qwen3InitializationSpec(
            policy_model_path="models/Qwen3-1.7B/policy",
            reference_model_path="models/Qwen3-1.7B/reference",
            tokenizer_path="models/Qwen3-1.7B/tokenizer",
        )

        def fake_exists(path: str) -> bool:
            return path != "models/Qwen3-1.7B/tokenizer"

        with self.assertRaisesRegex(ValueError, "tokenizer_path does not exist"):
            validate_qwen3_initialization_spec(
                spec=spec,
                require_existing_paths=True,
                path_exists=fake_exists,
            )

    def test_rejects_empty_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt at index 1 must be a non-empty string"):
            validate_prompt_tokenization(prompts=["valid", " "], encode=lambda text: [len(text)])

    def test_rejects_non_deterministic_tokenizer(self) -> None:
        tokenizer = _NonDeterministicTokenizer()
        with self.assertRaisesRegex(ValueError, "tokenizer output is non-deterministic"):
            validate_prompt_tokenization(prompts=["Solve this."], encode=tokenizer.encode)

    def test_rejects_non_integer_token_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokenizer output must be an integer token sequence"):
            validate_prompt_tokenization(prompts=["Solve this."], encode=lambda _text: [1, "x"])

    def test_rejects_invalid_training_stack_before_tokenization(self) -> None:
        config = TrainingStackConfig(base_model="Qwen3-8B")
        spec = Qwen3InitializationSpec(
            policy_model_path="models/Qwen3-1.7B/policy",
            reference_model_path="models/Qwen3-1.7B/reference",
            tokenizer_path="models/Qwen3-1.7B/tokenizer",
        )
        with self.assertRaisesRegex(ValueError, "base_model must be 'Qwen3-1.7B'"):
            validate_qwen3_stack_and_tokenization(
                config=config,
                spec=spec,
                prompts=["solve this"],
                encode=lambda text: [ord(char) for char in text],
            )


if __name__ == "__main__":
    unittest.main()
