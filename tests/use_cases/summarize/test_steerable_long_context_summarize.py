from intelligence_layer.core import Language, NoOpTracer
from intelligence_layer.core.model import LuminousControlModel
from intelligence_layer.use_cases import (
    LongContextSummarizeInput,
    SteerableLongContextSummarize,
)


def test_steerable_long_context_summarize_en(
    steerable_long_context_summarize: SteerableLongContextSummarize,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    output = steerable_long_context_summarize.run(input, NoOpTracer())

    assert output.partial_summaries
    assert any(
        "bear" in partial_summary.summary
        for partial_summary in output.partial_summaries
    )
    assert len(
        " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
    ) < len(long_text)


def test_steerable_long_context_summarize_adapts_to_instruction(
    luminous_control_model: LuminousControlModel,
    long_text: str,
) -> None:
    input = LongContextSummarizeInput(text=long_text)
    steerable_long_context_summarize_keyword = SteerableLongContextSummarize(
        max_generated_tokens=128,
        max_tokens_per_chunk=512,
        model=luminous_control_model,
        instruction_configs={Language("en"): "Summarize using bullet points."},
    )

    output = steerable_long_context_summarize_keyword.run(input, NoOpTracer())

    assert output.partial_summaries
    assert any(
        "bear" in partial_summary.summary
        for partial_summary in output.partial_summaries
    )
    assert all(
        partial_summary.summary.startswith("- ")
        for partial_summary in output.partial_summaries
    )
    assert len(
        " ".join(
            partial_summary.summary for partial_summary in output.partial_summaries
        )
    ) < len(long_text)
