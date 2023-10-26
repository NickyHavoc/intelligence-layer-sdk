from typing import Optional, Sequence

from aleph_alpha_client import Client
from pydantic import BaseModel

from intelligence_layer.core.complete import (
    Instruct,
    InstructInput,
    InstructOutput,
)
from intelligence_layer.core.prompt_template import (
    PromptWithMetadata,
)
from intelligence_layer.core.text_highlight import (
    TextHighlight,
    TextHighlightInput,
)
from intelligence_layer.core.task import Chunk, Task
from intelligence_layer.core.logger import DebugLogger


class SingleChunkQaInput(BaseModel):
    """The input for a `SingleChunkQa` task.

    Attributes:
        chunk: The (short) text to be asked about. Usually measures one or a few paragraph(s).
            Can't be longer than the context length of the model used minus the size of the system prompt.
        question: The question to be asked by about the chunk.
    """

    chunk: Chunk
    question: str


class SingleChunkQaOutput(BaseModel):
    """The output of a `SingleChunkQa` task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        highlights: Highlights indicating which parts of the chunk contributed to the answer.
            Each highlight is a quote from the text.
    """

    answer: Optional[str]
    highlights: Sequence[str]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    """Answer a question on the basis of one chunk.

    Uses Aleph Alpha models to generate a natural language answer for a text chunk given a question.
    Will answer `None` if the language model determines that the question cannot be answered on the
    basis of the text.

    Note:
        `model` provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question.
            Includes liquid logic interpreted by 'PromptTemplate' specifically for generating
            explainability-based highlights using `TextHighlight`.
        NO_ANSWER_STR: The string to be generated by the model in case no answer can be found.

    Example:
        >>> client = Client(os.getenv("AA_TOKEN"))
        >>> task = SingleChunkQa(client)
        >>> input = SingleChunkQaInput(
        >>>     chunk="Tina does not like pizza. However, Mike does.",
        >>>     question="Who likes pizza?"
        >>> )
        >>> logger = InMemoryLogger(name="Single Chunk QA")
        >>> output = task.run(input, logger)
        >>> print(output.answer)
        Mike likes pizza.
    """

    PROMPT_TEMPLATE_STR = """### Instruction:
{{question}}
If there's no answer, say "{{no_answer_text}}".

### Input:
{% promptrange text %}{{text}}{% endpromptrange %}

### Response:"""
    NO_ANSWER_STR = "NO_ANSWER_IN_TEXT"

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        super().__init__()
        self._client = client
        self._model = model
        self._instruction = Instruct(client)
        self._text_highlight = TextHighlight(client)

    def run(
        self, input: SingleChunkQaInput, logger: DebugLogger
    ) -> SingleChunkQaOutput:
        output = self._instruct(
            f"""{input.question}
If there's no answer, say "{self.NO_ANSWER_STR}".""",
            input.chunk,
            logger,
        )
        answer = self._no_answer_to_none(output.response.strip())
        highlights = (
            self._get_highlights(
                output.prompt_with_metadata,
                output.response,
                logger,
            )
            if answer
            else []
        )
        return SingleChunkQaOutput(
            answer=answer,
            highlights=highlights,
        )

    def _instruct(
        self, instruction: str, input: str, logger: DebugLogger
    ) -> InstructOutput:
        return self._instruction.run(
            InstructInput(instruction=instruction, input=input, model=self._model),
            logger,
        )

    def _get_highlights(
        self,
        prompt_with_metadata: PromptWithMetadata,
        completion: str,
        logger: DebugLogger,
    ) -> Sequence[str]:
        highlight_input = TextHighlightInput(
            prompt_with_metadata=prompt_with_metadata,
            target=completion,
            model=self._model,
            focus_ranges=frozenset({"input"}),
        )
        highlight_output = self._text_highlight.run(highlight_input, logger)
        return [h.text for h in highlight_output.highlights if h.score > 0]

    def _no_answer_to_none(self, completion: str) -> Optional[str]:
        return completion if completion != self.NO_ANSWER_STR else None
