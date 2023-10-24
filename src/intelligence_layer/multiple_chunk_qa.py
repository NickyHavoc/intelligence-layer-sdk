from typing import Iterable, Optional, Sequence
from intelligence_layer.single_chunk_qa import (
    SingleChunkQaInput,
    SingleChunkQaOutput,
    SingleChunkQa,
)
from intelligence_layer.task import Chunk, DebugLogger, Task
from aleph_alpha_client import (
    Client,
    Prompt,
)

from intelligence_layer.prompt_template import (
    PromptTemplate,
)
from intelligence_layer.completion import (
    Instruction,
    InstructionInput,
    InstructionOutput,
)
from pydantic import BaseModel


class MultipleChunkQaInput(BaseModel):
    """Input for a multiple chunk QA task.

    Attributes:
        chunks: list of chunks that will be used to answer the question.
            This can be an arbitrarily long list of chunks.
        question: The question that will be answered based on the chunks.
    """

    chunks: Sequence[Chunk]
    question: str


class Source(BaseModel):
    """Source for the multiple chunk QA output.

    Attributes:
        chunk: Piece of the original text that the qa output answer is based on.
        highlights: The specific sentences that explain the answer the most.
            These are generated by the TextHighlight Task.
    """

    chunk: Chunk
    highlights: Sequence[str]


class MultipleChunkQaOutput(BaseModel):
    """Multiple chunk qa output.

    Attributes:
        answer: The answer generated by the task. Can be a string or None (if no answer was found).
        sources: All the sources used to generate the answer.
    """

    answer: Optional[str]
    sources: Sequence[Source]


class MultipleChunkQa(Task[MultipleChunkQaInput, MultipleChunkQaOutput]):
    """Task implementation for answering a question based on multiple chunks.

    Uses Aleph Alpha models to generate a natural language answer based on multiple text chunks.
    Use this instead of SingleChunkQa if the texts you would like to ask about are larger than the model's context size.
    This task relies on SingleChunkQa to generate answers based on chunks and then merges the answers into a single final answer.

    Includes logic to return 'answer = None' if the language model determines that the question cannot be answered on the basis of the chunks.

    Note:
        'model' provided must be a control-type model for the prompt to function as expected.

    Args:
        client: Aleph Alpha client instance for running model related API calls.
        model: A valid Aleph Alpha model name.

    Attributes:
        PROMPT_TEMPLATE_STR: The prompt template used for answering the question.
            'chunk' and 'question' will be inserted here.

    Example:
        >>> client = Client(token="AA_TOKEN")
        >>> task = MultipleChunkQa(client)
        >>> input = MultipleChunkQaInput(
        >>>     chunks=["Tina does not like pizza.", "However, Mike does."],
        >>>     question="Who likes pizza?"
        >>> )
        >>> logger = InMemoryLogger(name="QA")
        >>> output = task.run(input, logger)
        >>> print(output.answer)
        Mike likes pizza.
    """

    INSTRUCTION = """You will be given a number of Answers to a Question. Based on them, generate a single final answer.
Condense multiple answers into a single answer. Rely only on the provided answers. Don't use the world's knowledge. The answer should combine the individual answers. If the answers contradict each other, e.g., one saying that the colour is green and the other saying that the colour is black, say that there are contradicting answers saying the colour is green or the colour is black."""

    def __init__(
        self,
        client: Client,
        model: str = "luminous-supreme-control",
    ):
        self._client = client
        self._instruction = Instruction(client)
        self._single_chunk_qa = SingleChunkQa(client, model)
        self._model = model

    def run(
        self, input: MultipleChunkQaInput, logger: DebugLogger
    ) -> MultipleChunkQaOutput:
        qa_outputs = self._single_chunk_qa.run_concurrently(
            (
                SingleChunkQaInput(question=input.question, chunk=chunk)
                for chunk in input.chunks
            ),
            logger,
        )

        final_answer = self._merge_answers(input.question, qa_outputs, logger)

        return MultipleChunkQaOutput(
            answer=final_answer,
            sources=[
                Source(chunk=chunk, highlights=qa_output.highlights)
                for qa_output, chunk in zip(qa_outputs, input.chunks)
                if qa_output.answer
            ],
        )

    def _merge_answers(
        self,
        question: str,
        qa_outputs: Iterable[SingleChunkQaOutput],
        logger: DebugLogger,
    ) -> Optional[str]:
        answers = [output.answer for output in qa_outputs if output.answer]
        if len(answers) == 0:
            return None
        elif len(answers) == 1:
            return answers[0]

        joined_answers = "\n".join(answers)
        return self._instruct(
            f"""Question: {question}

Answers:
{joined_answers}""",
            logger,
        ).response

    def _instruct(self, input: str, logger: DebugLogger) -> InstructionOutput:
        return self._instruction.run(
            InstructionInput(
                instruction=self.INSTRUCTION,
                input=input,
                model=self._model,
                response_prefix="\nFinal answer:",
            ),
            logger,
        )
