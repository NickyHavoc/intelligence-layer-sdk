from itertools import combinations
import math
from typing import Mapping, Sequence

from aleph_alpha_client import Prompt
from liquid import Template
from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.model import CompleteInput, CompleteOutput, ControlModel
from intelligence_layer.core.tracer.tracer import NoOpTracer, TaskSpan, Tracer
from intelligence_layer.evaluation.dataset.domain import Example
from intelligence_layer.evaluation.evaluation.evaluator.elo_evaluator import ComparisonEvaluation, EloGradingInput, MatchOutcome, Matches
from intelligence_layer.evaluation.evaluation.evaluator.incremental_evaluator import IncrementalEvaluationLogic
from intelligence_layer.evaluation.run.domain import SuccessfulExampleOutput
from intelligence_layer.examples.qa.single_chunk_qa import QA_INSTRUCTIONS, SingleChunkQaInput, SingleChunkQaOutput


class IncrementalEloQaEvaluationLogic(
    IncrementalEvaluationLogic[SingleChunkQaInput, SingleChunkQaOutput, SingleChunkQaOutput, Matches]
):
    INPUT_TEMPLATE = """
Your task is to compare two answers to an instruction on one metric.

Please make sure you read and understand these instruction carefully. Please keep this document open while reviewing, and refer to it as needed.

The Instruction for the answers was:{instruction}

Evaluation Procedure:
1. Read both answers carefully and identify the main facts and details they present.
2. Check if the answers contain any factual errors that are not supported by the instruction.
3. Evaluate which answer is more correct.

Answer A:{first_completion}

Answer B:{second_completion}

Which answer is more correct given the Instruction and Evaluation Procedure, Answer A or Answer B?

Response: Answer """
    VALUES = [
        " A",
        " B",
    ]  # The space before the A and B is important due to tokenization

    def __init__(
        self,
        model: ControlModel,
        #already_evaluated_outputs: list[list[SuccessfulExampleOutput[SingleChunkQaOutput]]],
        tracer: Tracer = NoOpTracer(),
    ):
        super().__init__()
        self._model = model
        self.tracer = tracer
        #self.already_evaluated_outputs = already_evaluated_outputs
        
    def do_incremental_evaluate(
        self,
        example: Example[SingleChunkQaInput, SingleChunkQaOutput],
        outputs: list[SuccessfulExampleOutput[SingleChunkQaOutput]],
        already_evaluated_outputs: list[list[SuccessfulExampleOutput[SingleChunkQaOutput]]],
    ) -> Matches:
        
        pairs = combinations(outputs, 2)
        print('PAIRS', pairs)
        unique_pre_evaluated_runs: set[str] = set()
       
        for pre_run_output in already_evaluated_outputs:
            for current_output in pre_run_output:
                unique_pre_evaluated_runs.add(current_output.run_id)

        
        return Matches(
            comparison_evaluations=[
                ComparisonEvaluation(
                    first_player=player_a.run_id,
                    second_player=player_b.run_id,
                    outcome=self.grade(player_a, player_b, example),
                )
                for [player_a, player_b] in pairs
                if unique_pre_evaluated_runs is None
                or len(unique_pre_evaluated_runs) == 0
                or not (player_a.run_id in unique_pre_evaluated_runs and player_b.run_id in unique_pre_evaluated_runs)              
            ]  
        )
    
    

    def grade(
        self,
        first: SuccessfulExampleOutput[SingleChunkQaOutput],
        second: SuccessfulExampleOutput[SingleChunkQaOutput],
        example: Example[SingleChunkQaInput, SingleChunkQaOutput],
    ) -> MatchOutcome:
        grading_input = self._create_grading_input(first, second, example)

        print('PLAYER A: ', first.run_id)
        print('PLAYER B: ', second.run_id)
        print('example_id: ', example.id)
        print('______________________')
        return MatchOutcome(
            self.do_run(
                grading_input,
                self.tracer.task_span(
                    task_name="elo_qa_run_grader", input=grading_input
                ),
            )
        ) 
    
    def _create_grading_input(
        self,
        first: SuccessfulExampleOutput[SingleChunkQaOutput],
        second: SuccessfulExampleOutput[SingleChunkQaOutput],
        example: Example[SingleChunkQaInput, SingleChunkQaOutput],
    ) -> EloGradingInput:
        qa_instruction = Template(
            QA_INSTRUCTIONS[Language("en")].unformatted_instruction
        ).render(question=example.input.question)

        no_answer = "There is no answer."
        return EloGradingInput(
            instruction=f"{example.input.chunk} {qa_instruction}",
            first_completion=(
                first.output.answer if first.output.answer is not None else no_answer
            ),
            second_completion=(
                second.output.answer if second.output.answer is not None else no_answer
            ),
        )

    def do_run(self, input: EloGradingInput, task_span: TaskSpan) -> MatchOutcome:
        text = self.INPUT_TEMPLATE.format(
            instruction=input.instruction,
            first_completion=input.first_completion,
            second_completion=input.second_completion,
        )

        complete_input = CompleteInput(
            prompt=Prompt.from_text(text),
            maximum_tokens=1,
            log_probs=3,
            disable_optimizations=True,
        )
        complete_output = self._model.complete_task().run(complete_input, task_span)

        return self.calculate_winners(complete_output)
    
    def calculate_winners(self, complete_output: CompleteOutput) -> MatchOutcome:
        default_log_prob = float("-inf")

        def get_normalized_prob(
            log_prob_list: Sequence[Mapping[str, float | None]] | None,
        ) -> float:
            assert log_prob_list is not None
            log_probs = log_prob_list[0]
            values = [
                math.exp(log_probs.get(str(key), default_log_prob) or default_log_prob)
                for key in self.VALUES
            ]
            if all(v == 0 for v in values):
                raise ValueError(
                    f"LLM evaluation response does not contain logprobs for the required tokens for the values: {self.VALUES}"
                )
            return values[0] / sum(values)

        def categorize_value(value: float) -> MatchOutcome:
            if value > 0.7:
                return MatchOutcome.A_WINS
            elif 0.3 > value:
                return MatchOutcome.B_WINS
            else:
                return MatchOutcome.DRAW

        normalized_probability = get_normalized_prob(
            complete_output.completions[0].log_probs
        )
        return categorize_value(normalized_probability)
