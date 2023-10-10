import statistics
from typing import Iterable, Sequence

from aleph_alpha_client import (
    Client,
    ExplanationRequest,
    ExplanationResponse,
    PromptGranularity,
    Text,
    TextPromptItemExplanation,
    Prompt,
)
from aleph_alpha_client.explanation import TextScoreWithRaw
from pydantic import BaseModel

from intelligence_layer.prompt_template import (
    PromptRange,
    PromptWithMetadata,
    TextCursor,
)
from intelligence_layer.task import DebugLog, LogLevel, Task


class TextHighlightInput(BaseModel):
    """Input for a highlight task"""

    prompt_with_metadata: PromptWithMetadata
    target: str
    model: str


class ScoredTextHighlight(BaseModel):
    text: str
    score: float


class TextHighlightOutput(BaseModel):
    """Output of for a highlight task"""

    highlights: Sequence[ScoredTextHighlight]
    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


class TextHighlight(Task[TextHighlightInput, TextHighlightOutput]):
    client: Client

    def __init__(self, client: Client, log_level: LogLevel) -> None:
        """Initializes the Task.

        Args:
        - client: the aleph alpha client
        """
        super().__init__()
        self.client = client
        self.log_level = log_level

    def run(self, input: TextHighlightInput) -> TextHighlightOutput:
        debug_log = DebugLog.enabled(level=self.log_level)
        explanation = self._explain(
            prompt=input.prompt_with_metadata.prompt,
            target=input.target,
            model=input.model,
            debug_log=debug_log,
        )
        prompt_ranges = self._flatten_prompt_ranges(
            input.prompt_with_metadata.ranges.values()
        )
        text_prompt_item_explanations_and_indices = (
            self._extract_text_prompt_item_explanations_and_item_index(
                input.prompt_with_metadata.prompt, explanation
            )
        )
        highlights = self._to_highlights(
            prompt_ranges,
            text_prompt_item_explanations_and_indices,
            debug_log,
        )
        return TextHighlightOutput(highlights=highlights, debug_log=debug_log)

    def _explain(
        self, prompt: Prompt, target: str, model: str, debug_log: DebugLog
    ) -> ExplanationResponse:
        request = ExplanationRequest(
            prompt,
            target,
            prompt_granularity=PromptGranularity.Sentence,
        )
        response = self.client.explain(request, model)
        debug_log.debug(
            "Explanation Request/Response", {"request": request, "response": response}
        )
        return response

    def _flatten_prompt_ranges(
        self, prompt_ranges: Iterable[Sequence[PromptRange]]
    ) -> Sequence[PromptRange]:
        return [pr for prs in prompt_ranges for pr in prs]

    def _extract_text_prompt_item_explanations_and_item_index(
        self,
        prompt: Prompt,
        explanation_response: ExplanationResponse,
    ) -> Sequence[tuple[TextPromptItemExplanation, int]]:
        prompt_texts_and_indices = [
            (prompt_text, idx)
            for idx, prompt_text in enumerate(prompt.items)
            if isinstance(prompt_text, Text)
        ]
        text_prompt_item_explanations = [
            explanation
            for explanation in explanation_response.explanations[0].items
            if isinstance(explanation, TextPromptItemExplanation)
        ]  # explanations[0], because one explanation for each target
        assert len(prompt_texts_and_indices) == len(text_prompt_item_explanations)
        return [
            (
                text_prompt_item_explanation.with_text(prompt_text_and_index[0]),
                prompt_text_and_index[1],
            )
            for prompt_text_and_index, text_prompt_item_explanation in zip(
                prompt_texts_and_indices, text_prompt_item_explanations
            )
        ]

    @staticmethod
    def _is_within_prompt_range(
        prompt_range: PromptRange,
        item_check: int,
        pos_check: int,
    ) -> bool:
        assert isinstance(prompt_range.start, TextCursor)
        assert isinstance(prompt_range.end, TextCursor)
        if item_check < prompt_range.start.item or item_check > prompt_range.end.item:
            return False
        elif (
            item_check == prompt_range.start.item
            and pos_check < prompt_range.start.position
        ):
            return False
        elif (
            item_check == prompt_range.end.item
            and pos_check > prompt_range.end.position
        ):
            return False
        return True

    @classmethod
    def _prompt_range_overlaps_with_text_score(
        cls,
        prompt_range: PromptRange,
        text_score: TextScoreWithRaw,
        explanation_item_idx: int,
    ) -> bool:
        return cls._is_within_prompt_range(
            prompt_range,
            explanation_item_idx,
            text_score.start,
        ) or cls._is_within_prompt_range(
            prompt_range,
            explanation_item_idx,
            text_score.start + text_score.length - 1,
        )

    def _to_highlights(
        self,
        prompt_ranges: Sequence[PromptRange],
        text_prompt_item_explanations_and_indices: Sequence[
            tuple[TextPromptItemExplanation, int]
        ],
        debug_log: DebugLog,
    ) -> Sequence[ScoredTextHighlight]:
        overlapping_and_flat = [
            text_score
            for text_prompt_item_explanation, explanation_idx in text_prompt_item_explanations_and_indices
            for text_score in text_prompt_item_explanation.scores
            if isinstance(text_score, TextScoreWithRaw)
            and any(
                self._prompt_range_overlaps_with_text_score(
                    prompt_range, text_score, explanation_idx
                )
                for prompt_range in prompt_ranges
            )
        ]
        debug_log.info(
            "Explanation scores",
            [
                {
                    "text": text_score.text,
                    "score": text_score.score,
                }
                for text_score in overlapping_and_flat
            ],
        )
        z_scores = self._z_scores([s.score for s in overlapping_and_flat], debug_log)
        scored_highlights = [
            ScoredTextHighlight(text=text_score.text, score=z_score)
            for text_score, z_score in zip(overlapping_and_flat, z_scores)
        ]
        debug_log.info("Unfiltered highlights", scored_highlights)
        return self._filter_highlights(scored_highlights)

    @staticmethod
    def _z_scores(data: Sequence[float], debug_log: DebugLog) -> Sequence[float]:
        mean = statistics.mean(data)
        stdev = (
            statistics.stdev(data) if len(data) > 1 else 0
        )  # standard deviation not defined for n < 2
        debug_log.info("Highlight statistics", {"mean": mean, "std_dev": stdev})
        return [((x - mean) / stdev if stdev > 0 else 0) for x in data]

    def _filter_highlights(
        self,
        scored_highlights: Sequence[ScoredTextHighlight],
        z_score_limit: float = 0.5,
    ) -> Sequence[ScoredTextHighlight]:
        return [h for h in scored_highlights if abs(h.score) >= z_score_limit]
