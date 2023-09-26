from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    Protocol,
    runtime_checkable,
)
from aleph_alpha_client import CompletionRequest, CompletionResponse, Prompt
from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
)
from typing_extensions import TypeAliasType


if TYPE_CHECKING:
    PydanticSerializable = (
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
        | Prompt
        | CompletionRequest
        | CompletionResponse
    )
else:
    PydanticSerializable = TypeAliasType(
        "PydanticSerializable",
        int
        | float
        | str
        | Sequence["PydanticSerializable"]
        | Mapping[str, "PydanticSerializable"]
        | None
        | bool
        | BaseModel
        | Prompt
        | CompletionRequest
        | CompletionResponse,
    )

LogLevel = Literal["info", "debug"]


class LogEntry(BaseModel):
    message: str
    level: LogLevel
    value: SerializeAsAny[PydanticSerializable]


class DebugLog(BaseModel):
    """Provides key steps, decisions, and intermediate outputs of a task's process."""

    level: LogLevel = Field(exclude=True)
    log: list[LogEntry] = []

    def info(self, message: str, value: PydanticSerializable) -> None:
        self.log.append(LogEntry(message=message, level="info", value=value))

    def debug(self, message: str, value: PydanticSerializable) -> None:
        if self.level == "debug":
            self.log.append(LogEntry(message=message, level="debug", value=value))


@runtime_checkable
class OutputProtocol(Protocol):
    """Minimum interface for a `Task`'s output."""

    debug_log: DebugLog
    """Provides key steps, decisions, and intermediate outputs of a task's process."""


Input = TypeVar("Input")
"""Interface to be passed to the task with all data needed to run the process.
Ideally, these are specified in terms related to the use-case, rather than lower-level
configuration options."""
Output = TypeVar("Output", bound=OutputProtocol)
"""Interface of the output returned by the task.
It is required to adhere to the `OutputProtocol` and provide a `DebugLog` of key steps
and decisions made in the process of generating the output."""


class Task(Generic[Input, Output]):
    """Base task interface. This may consist of several sub-tasks to accomplish the given task.

    Generics:
        Input: Interface to be passed to the task with all data needed to run the process.
            Ideally, these are specified in terms related to the use-case, rather than lower-level
            configuration options.
        Output: Interface of the output returned by the task.
            It is required to adhere to the `OutputProtocol` and provide a `DebugLog` of key steps
            and decisions made in the process of generating the output.
    """

    @abstractmethod
    def run(self, input: Input) -> Output:
        """Executes the process for this use-case."""
        raise NotImplementedError
