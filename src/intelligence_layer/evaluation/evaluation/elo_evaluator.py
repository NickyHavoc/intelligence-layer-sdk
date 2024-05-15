from abc import abstractmethod
from itertools import combinations
from typing import Sequence, final

from pydantic import BaseModel

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation import EvaluationLogic
from intelligence_layer.evaluation.aggregation.elo import MatchOutcome
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput


class Match(BaseModel):
    player_a: str
    player_b: str
    outcome: MatchOutcome


class Matches(BaseModel):
    matches: Sequence[Match]


class EloGradingInput(BaseModel):
    instruction: str
    first_completion: str
    second_completion: str


class EloEvaluationLogic(EvaluationLogic[Input, Output, ExpectedOutput, Matches]):
    """Evaluation logic for a pair-wise ELO comparison.

    Args:
        grader: The :class:`Task` that perform the grading, i.e. the actual comparison of two run outputs.
        tracer: :class:`Tracer` for tracking and debugging

    """

    @final
    def do_evaluate(
        self,
        example: Example[Input, ExpectedOutput],
        *output: SuccessfulExampleOutput[Output],
    ) -> Matches:
        pairs = combinations(output, 2)
        return Matches(
            matches=[
                Match(
                    player_a=first.run_id,
                    player_b=second.run_id,
                    outcome=self.grade(first, second, example),
                )
                for [first, second] in pairs
            ]
        )

    @abstractmethod
    def grade(
        self,
        output_a: SuccessfulExampleOutput[Output],
        output_b: SuccessfulExampleOutput[Output],
        example: Example[Input, ExpectedOutput],
    ) -> MatchOutcome:
        pass
