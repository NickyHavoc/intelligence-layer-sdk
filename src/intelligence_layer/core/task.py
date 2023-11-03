from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import functools
from itertools import islice
from typing import (
    Any,
    Generic,
    Iterable,
    NewType,
    Sequence,
    TypeVar,
    Callable,
)

from pydantic import (
    BaseModel,
)

from intelligence_layer.core.logger import DebugLogger, PydanticSerializable


Chunk = NewType("Chunk", str)
"""Segment of a larger text.

This type infers that the string is smaller than the context size of the model where it is used.

LLMs can't process documents larger than their context size.
To handle this, documents have to be split up into smaller segments that fit within their context size.
These smaller segments are referred to as chunks.
"""

LogProb = NewType("LogProb", float)
Probability = NewType("Probability", float)


class Token(BaseModel):
    """A token class containing it's id and the raw token.

    This is used instead of the Aleph Alpha client Token class since this one is serializable,
    while the one from the client is not.
    """

    token: str
    token_id: int


Input = TypeVar("Input", bound=PydanticSerializable)
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=PydanticSerializable)
"""Interface of the output returned by the task."""


MAX_CONCURRENCY = 20
global_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)


class Task(ABC, Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.

        Output: Interface of the output returned by the task.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Decorates run method to auto log input and output for the task"""
        super().__init_subclass__(**kwargs)

        def log_run_input_output(
            func: Callable[["Task[Input, Output]", Input, DebugLogger], Output]
        ) -> Callable[["Task[Input, Output]", Input, DebugLogger], Output]:
            @functools.wraps(func)
            def inner(
                self: "Task[Input, Output]",
                input: Input,
                logger: DebugLogger,
            ) -> Output:
                with logger.task_span(type(self).__name__, input) as task_span:
                    output = func(self, input, task_span)
                    task_span.record_output(output)
                    return output

            return inner

        cls.run = log_run_input_output(cls.run)  # type: ignore

    @abstractmethod
    def run(self, input: Input, logger: DebugLogger) -> Output:
        """Executes the implementation of run for this use case.

        This takes an input and runs the implementation to generate an output.
        It takes a `DebugLogger` for tracing of the process.
        The Input and Output are logged by default.

        Args:
            input: Generic input defined by the task implementation
            logger: The `DebugLogger` used for tracing.
        Returns:
            Generic output defined by the task implementation.
        """
        ...

    def run_concurrently(
        self,
        inputs: Iterable[Input],
        debug_logger: DebugLogger,
        concurrency_limit: int = MAX_CONCURRENCY,
    ) -> Sequence[Output]:
        """Executes multiple processes of this task concurrently.

        Each provided input is potentially executed concurrently to the others. There is a global limit
        on the number of concurrently executed tasks that is shared by all tasks of all types.

        Args:
            inputs: The inputs that are potentially processed concurrently.
            debug_logger: The logger passed on the `run` method when executing a task.
            concurrency_limit: An optional additional limit for the number of concurrently executed task for
                this method call. This can be used to prevent queue-full or similar error of downstream APIs
                when the global concurrency limit is too high for a certain task.
        Returns:
            The `Output`\ s generated by calling `run` for each given `Input`.
            The order of `Output`\ s corresponds to the order of the `Input`\ s.
        """

        with debug_logger.span(f"Concurrent {type(self).__name__} tasks") as span:

            def run_batch(inputs: Iterable[Input]) -> Iterable[Output]:
                return global_executor.map(
                    lambda input: self.run(input, span),
                    inputs,
                )

            return [
                output
                for batch in batched(inputs, concurrency_limit)
                for output in run_batch(batch)
            ]


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[Iterable[T]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
