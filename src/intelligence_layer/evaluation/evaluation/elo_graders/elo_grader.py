from abc import abstractmethod
from typing import Generic, Sequence

from pydantic import BaseModel

from intelligence_layer.core.task import Input, Output
from intelligence_layer.evaluation import MatchOutcome
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


class EloGrader(
    Generic[
        Input,
        Output,
        ExpectedOutput,
    ],
):
    @abstractmethod
    def grade(
        self,
        output_a: SuccessfulExampleOutput[Output],
        output_b: SuccessfulExampleOutput[Output],
        example: Example[Input, ExpectedOutput],
    ) -> MatchOutcome:
        pass
