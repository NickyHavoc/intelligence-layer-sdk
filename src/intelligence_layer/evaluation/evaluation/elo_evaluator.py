from itertools import combinations

from intelligence_layer.core import Input, Output
from intelligence_layer.evaluation import EvaluationLogic
from intelligence_layer.evaluation.dataset.domain import Example, ExpectedOutput
from intelligence_layer.evaluation.evaluation.elo_graders.elo_grader import (
    EloGrader,
    Match,
    Matches,
)
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput


class EloEvaluationLogic(EvaluationLogic[Input, Output, ExpectedOutput, Matches]):
    """Evaluation logic for a pair-wise ELO comparison.

    Args:
        grader: The :class:`Task` that perform the grading, i.e. the actual comparison of two run outputs.
        tracer: :class:`Tracer` for tracking and debugging

    """

    def __init__(
        self,
        grader: EloGrader[Input, Output, ExpectedOutput],
    ):
        self._grader = grader

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
                    outcome=self._grader.grade(first, second, example),
                )
                for [first, second] in pairs
            ]
        )
