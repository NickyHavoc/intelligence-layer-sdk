from aleph_alpha_client import Client, Image
from pytest import fixture, raises
from intelligence_layer.core.text_highlight import TextHighlight, TextHighlightInput
from intelligence_layer.core.prompt_template import PromptTemplate

from intelligence_layer.core.logger import NoOpDebugLogger


@fixture
def text_highlight(client: Client) -> TextHighlight:
    return TextHighlight(client)


def test_text_highlight(text_highlight: TextHighlight) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?{% promptrange r1 %} Explanation should only highlight this. Latin name: Ursus Arctos.{% endpromptrange %} This should also not be highlighted.
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata()
    completion = " Ursus Arctos"
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata,
        target=completion,
        model=model,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    top_highlight = next(h for h in output.highlights if "Ursus" in h.text)
    assert all(
        top_highlight.score >= highlight.score for highlight in output.highlights
    )


def test_text_highlight_with_range_without_highlight(
    text_highlight: TextHighlight,
) -> None:
    answer = "Ursus Arctos"
    prompt_template_str = """Question: What is the Latin name of the brown bear?
{% promptrange no_content %}This is an unrelated sentence. And here is another one.{% endpromptrange %}
{% promptrange content %}Latin name: {{answer}}.{% endpromptrange %}
Answer:"""
    prompt_with_metadata = PromptTemplate(prompt_template_str).to_prompt_with_metadata(
        answer=answer
    )

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata,
        target=f" {answer}",
        model="luminous-base",
        focus_ranges=frozenset(["no_content"]),
    )
    output = text_highlight.run(input, NoOpDebugLogger())
    assert all(answer not in highlight.text for highlight in output.highlights)


def test_text_highlight_with_only_one_sentence(text_highlight: TextHighlight) -> None:
    prompt_template_str = """What is the Latin name of the brown bear? The answer is Ursus Arctos.{% promptrange r1 %} Explanation should not highlight anything.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata()
    completion = " Ursus Arctos"
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata,
        target=completion,
        model=model,
        focus_ranges=frozenset({"r1"}),
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert not output.highlights


def test_text_highlight_with_image_prompt(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: {% promptrange question %}What is the Latin name of the brown bear?{% endpromptrange %}
Text: {% promptrange text %}The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Here is an image, just for LOLs: {{image}}{range}abc{{image}}{range}
{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)


def test_text_highlight_without_range(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?
Text: The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
Here is an image, just for LOLs: {{image}}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata, target=completion, model=model
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)


def test_text_highlight_without_focus_range_highlights_entire_prompt(
    text_highlight: TextHighlight, prompt_image: Image
) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?
Text: The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.
{% promptrange text %}Here is an image, just for LOLs: {{image}}{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata(
        image=template.placeholder(prompt_image)
    )
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata,
        target=completion,
        model=model,
        focus_ranges=frozenset(),
    )
    output = text_highlight.run(input, NoOpDebugLogger())

    assert output.highlights
    assert any("bear" in highlight.text.lower() for highlight in output.highlights)


def test_text_highlight_with_unknown_range_raises(
    text_highlight: TextHighlight,
) -> None:
    prompt_template_str = """Question: What is the Latin name of the brown bear?
{% promptrange text %}Text: The brown bear (Ursus arctos) is a large bear species found across Eurasia and North America.{% endpromptrange %}
Answer:"""
    template = PromptTemplate(prompt_template_str)
    prompt_with_metadata = template.to_prompt_with_metadata()
    completion = " The latin name of the brown bear is Ursus arctos."
    model = "luminous-base"

    unknown_range_name = "bla"
    input = TextHighlightInput(
        prompt_with_metadata=prompt_with_metadata,
        target=completion,
        model=model,
        focus_ranges=frozenset([unknown_range_name]),
    )
    with raises(ValueError) as e:
        text_highlight.run(input, NoOpDebugLogger())

    assert unknown_range_name in str(e.value)
