from pydantic import BaseModel

from intelligence_layer.core.detect_language import Language
from intelligence_layer.core.chunk import Chunk


class LongContextSummarizeInput(BaseModel):
    """The input for a summarize-task for texts of any length.

    Attributes:
        text: A text of any length.
    """

    text: str


class SingleChunkSummarizeInput(BaseModel):
    """The input for a summarize-task that only deals with a single chunk.

    Attributes:
        chunk: The text chunk to be summarized.
        language: The desired language of the summary.
    """

    chunk: Chunk
    language: Language


class SummarizeOutput(BaseModel):
    """The output of a `Summarize` task.

    Attributes:
        summary: The summary generated by the task.
    """

    summary: str
