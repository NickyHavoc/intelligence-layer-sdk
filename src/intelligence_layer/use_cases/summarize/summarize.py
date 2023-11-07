from typing import Sequence

from pydantic import BaseModel

from intelligence_layer.core.chunk import Chunk
from intelligence_layer.core.detect_language import Language


class LongContextSummarizeInput(BaseModel):
    """The input for a summarize-task for a text of any length.

    Attributes:
        text: A text of any length.
    """

    text: str


class PartialSummary(BaseModel):
    summary: str
    chunk: Chunk


class LongContextSummarizeOutput(BaseModel):
    """The output of a summarize-task for a text of any length.

    Attributes:
        partial_summaries: Chunk-wise summaries.
    """

    partial_summaries: Sequence[PartialSummary]


class SingleChunkSummarizeInput(BaseModel):
    """The input for a summarize-task that only deals with a single chunk.

    Attributes:
        chunk: The text chunk to be summarized.
        language: The desired language of the summary. ISO 619 str with language e.g. en, fr, etc.
    """

    chunk: Chunk
    language: Language


class SingleChunkSummarizeOutput(BaseModel):
    """The input of a summarize-task that only takes a single chunk.

    Attributes:
        summary: The summary generated by the task.
    """

    summary: str
