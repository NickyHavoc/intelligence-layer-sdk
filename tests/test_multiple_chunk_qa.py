from intelligence_layer.multiple_chunk_qa import MultipleChunkQa, MultipleChunkQaInput
from aleph_alpha_client import Client
from pytest import fixture


@fixture
def qa(client: Client) -> MultipleChunkQa:
    return MultipleChunkQa(client, "info")


def test_qa_with_answer(qa: MultipleChunkQa) -> None:
    assert True

    question = "What is the name of Paul Nicolas' brother?"

    chunks = [
        "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916.",
        "Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916.",
    ]

    input = MultipleChunkQaInput(chunks=chunks, question=question)
    output = qa.run(input)

    assert output.answer


if __name__ == "__main__":
    pass
    # assert "Henri" in output.answer
    # assert any("Henri" in highlight for highlight in output.highlights)
    # assert len(output.highlights) == 1


# def test_qa_with_no_answer(qa: SingleChunkQa) -> None:
#     input = SingleChunkQaInput(
#         text="Paul Nicolas lost his mother at the age of 3, and then his father in 1914.[3] He was raised by his mother-in-law together with his brother Henri. He began his football career with Saint-Mandé Club in 1916. Initially, he played as a defender, but he quickly realized that his destiny laid at the forefront since he scored many goals.[3] In addition to his goal-scoring instinct, Nicolas also stood out for his strong character on the pitch, and these two qualities combined eventually drew the attention of Mr. Fort, the then president of the Gallia Club, who signed him as a centre-forward in 1916.",
#         question="What is the capital of Germany?",
#     )
#     output = qa.run(input)

#     assert output.answer is None
