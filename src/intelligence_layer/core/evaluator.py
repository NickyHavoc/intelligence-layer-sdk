from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, Mapping, Optional, Sequence, TypeVar
from uuid import uuid4

import nltk  # type: ignore
from nltk.tokenize import RegexpTokenizer  # type: ignore
from nltk.translate.bleu_score import sentence_bleu  # type: ignore
from pydantic import BaseModel, Field
from rouge import Rouge  # type: ignore
from tqdm import tqdm

from intelligence_layer.core.task import Input
from intelligence_layer.core.tracer import PydanticSerializable, Tracer

nltk.download("punkt")

ExpectedOutput = TypeVar("ExpectedOutput", bound=PydanticSerializable)
Evaluation = TypeVar("Evaluation", bound=PydanticSerializable)
AggregatedEvaluation = TypeVar("AggregatedEvaluation", bound=PydanticSerializable)


class Example(BaseModel, Generic[Input, ExpectedOutput]):
    """Example case used for evaluations.

    Attributes:
        input: Input for the task. Has to be same type as the input for the task used.
        expected_output: The expected output from a given example run.
            This will be used by the evaluator to compare the received output with.
        ident: Identifier for the example, defaults to uuid.
    """

    input: Input
    expected_output: ExpectedOutput
    ident: Optional[str] = Field(default_factory=lambda: str(uuid4()))


class Dataset(BaseModel, Generic[Input, ExpectedOutput]):
    """A dataset of examples used for evaluation of a task.

    Attributes:
        name: This a human readable identifier for a dataset.
        examples: The actual examples that a task will be evaluated on.
    """

    name: str
    examples: Sequence[Example[Input, ExpectedOutput]]


class Evaluator(ABC, Generic[Input, ExpectedOutput, Evaluation, AggregatedEvaluation]):
    """Base evaluator interface. This should run certain evaluation steps for some job.

    We suggest supplying a `Task` in the `__init__` method and running it in the `evaluate` method.

    Generics:
        Input: Interface to be passed to the task that shall be evaluated.
        ExpectedOutput: Output that is expected from the task run with the supplied input.
        Evaluation: Interface of the metrics that come from the evaluated task.
        AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
    """

    @abstractmethod
    def evaluate(
        self,
        input: Input,
        tracer: Tracer,
        expected_output: ExpectedOutput,
    ) -> Evaluation:
        """Executes the evaluation for this use-case.

        The implementation of this method is responsible for running a task (usually supplied by the __init__ method)
        and making any comparisons relevant to the evaluation.
        Based on the results, it should create an `Evaluation` class with all the metrics and return it.

        Args:
            input: Interface to be passed to the task that shall be evaluated.
            tracer: Ttracer used for tracing of tasks.
            expected_output: Output that is expected from the task run with the supplied input.
        Returns:
            Evaluation: interface of the metrics that come from the evaluated task.
        """
        pass

    def evaluate_dataset(
        self, dataset: Dataset[Input, ExpectedOutput], tracer: Tracer
    ) -> AggregatedEvaluation:
        """Evaluates an entire datasets in a threaded manner and aggregates the results into an `AggregatedEvaluation`.

        This will call the `run` method for each example in the dataset.
        Finally, it will call the `aggregate` method and return the aggregated results.

        Args:
            dataset: Dataset that will be used to evaluate a task.
            tracer: tracer used for tracing.
        Returns:
            AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            evaluations = list(
                tqdm(
                    executor.map(
                        lambda idx_example: self.evaluate(
                            idx_example.input,
                            tracer,
                            idx_example.expected_output,
                        ),
                        dataset.examples,
                    ),
                    total=len(dataset.examples),
                    desc="Evaluating",
                )
            )
        return self.aggregate(evaluations)

    @abstractmethod
    def aggregate(self, evaluations: Sequence[Evaluation]) -> AggregatedEvaluation:
        """`Evaluator`-specific method for aggregating individual `Evaluations` into report-like `Aggregated Evaluation`.

        This method is responsible for taking the results of an evaluation run and aggregating all the results.
        It should create an `AggregatedEvaluation` class and return it at the end.

        Args:
            evalautions: The results from running `evaluate_dataset` with a task.
        Returns:
            AggregatedEvaluation: The aggregated results of an evaluation run with a dataset.
        """
        pass


def tokenize(input: str) -> Sequence[str]:
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(input.lower())
    assert isinstance(tokens, list)
    return tokens


def calculate_bleu(hypothesis: str, reference: str) -> float:
    hypothesis_tokens = tokenize(hypothesis)
    reference_tokens = tokenize(reference)
    bleu_score = sentence_bleu(
        references=[reference_tokens], hypothesis=hypothesis_tokens
    )
    return bleu_score if isinstance(bleu_score, float) else 0.0


@dataclass
class RougeScores:
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_rouge_results(cls, rouge_results: Mapping[str, float]) -> "RougeScores":
        return cls(
            precision=rouge_results["p"],
            recall=rouge_results["r"],
            f1=rouge_results["f"],
        )


def calculate_rouge(hypothesis: str, reference: str) -> RougeScores:
    hypothesis = " ".join(tokenize(hypothesis))
    reference = " ".join(tokenize(reference))
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference)[0]["rouge-2"]
    return RougeScores.from_rouge_results(rouge_scores)
