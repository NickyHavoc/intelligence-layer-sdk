from typing import Optional, Sequence
from aleph_alpha_client import (
    Client,
)
from pydantic import BaseModel

from intelligence_layer.completion import (
    Instruction,
    InstructionInput,
    InstructionOutput,
)
from intelligence_layer.text_highlight import (
    TextHighlight,
    TextHighlightInput,
)
from intelligence_layer.prompt_template import (
    PromptWithMetadata,
)
from intelligence_layer.task import (
    Chunk,
    DebugLogger,
    Task,
)


class SingleChunkQaInput(BaseModel):
    """The input for a single chunk QA task.

    Attributes:
        chunk: The (short) text to be asked about. Usually measures one or a few paragraph(s).
            Can't be longer than the context length of the model used minus the size of the system prompt.
        question: The question being asked.
    """

    chunk: Chunk
    question: str


class SingleChunkQaOutput(BaseModel):
    """The output of a single chunk QA task.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        highlights: Highlights indicating which parts of the chunk contributed to the answer. Each highlight is a quote from the text.
    """

    answer: Optional[str]
    highlights: Sequence[str]


class SingleChunkQa(Task[SingleChunkQaInput, SingleChunkQaOutput]):
    """Task implementation for answering a question based on a single chunk.

    Depends on SingleChunkQaInput and SingleChunkQaOutput. Uses Aleph Alpha models to generate a natural language answer for a text chunk.

    Includes logic to return 'answer = None' if the language model determines that the question cannot be answered on the basis of the text.

    Note:
        'model' provided should be a control-type model.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question. 'chunk' and 'question' will be inserted here. Includes liquid logic interpreted by 'PromptTemplate':
        NO_ANSWER_STR: The string to be generated by the model in case no answer can be found.

    Example:
        >>> client = Client(token="YOUR_AA_TOKEN")
        >>> task = SingleChunkQa(client)
        >>> input = SingleChunkQaInput(
        >>>     chunk="Tina does not like pizza. However, Mike does.",
        >>>     question="Who likes pizza?"
        >>> )
        >>> logger = InMemoryLogger(name="QA")
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
        self._client = client
        self._model = model
        self._instruction = Instruction(client)
        self._text_highlight = TextHighlight(client)

    def run(
        self, input: SingleChunkQaInput, logger: DebugLogger
    ) -> SingleChunkQaOutput:
        output = self._instruct(
            f"""{input.question}
If there's no answer, say "{self.NO_ANSWER_STR}".""",
            input.chunk,
            logger.child_logger("Generate Answer"),
        )
        highlights = self._get_highlights(
            output.prompt_with_metadata,
            output.response,
            logger.child_logger("Explain Answer"),
        )
        return SingleChunkQaOutput(
            answer=self._no_answer_to_none(output.response.strip()),
            highlights=highlights,
        )

    def _instruct(
        self, instruction: str, input: str, logger: DebugLogger
    ) -> InstructionOutput:
        return self._instruction.run(
            InstructionInput(instruction=instruction, input=input, model=self._model),
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
