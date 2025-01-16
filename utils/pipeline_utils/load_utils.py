from typing import Any, List
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from src.utils.misc import hash_value
import src.utils.custom_prompts as cp
from openai import BadRequestError
from langchain_community.vectorstores import faiss
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

from distilabel.pipeline import Pipeline
from distilabel.llms.azure import AzureOpenAILLM
from distilabel.steps.tasks.evol_instruct.base import EvolInstruct
import json
import random


def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.
    """
    l, r = 0, len(s) - 1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l : min(r, len(s))]


def generate_instructions_gen(
    chunk: Any, number_of_questions: int = 5, client: AzureChatOpenAI = None
) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types
    `pdf`, `json`, or `txt`.
    """
    language = map_language_flags(chunk["language"])
    try:
        response = client(
            messages=[
                SystemMessage(
                    content=cp.QUESTION_GEN_PROMPT.format(
                        number_of_questions=number_of_questions
                    ),
                ),
                HumanMessage(content=cp.QUESTION_GEN_QUESTION_EXAMPLE_COMPILED_1),
                AIMessage(content=cp.QUESTION_GEN_ANSWER_EXAMPLE_1),
                HumanMessage(
                    content=cp.QUESTION_GEN_QUESTION_HUMAN_MESSAGE.format(
                        url=chunk["url"],
                        number_of_questions=number_of_questions,
                        language=language,
                        chunk=chunk["chunk"],
                    )
                ),
            ],
        )
        queries = response.content.split("\n")
    except BadRequestError:
        queries = []

    return queries


def encode_question_gen(
    question: str, embeddings: AzureOpenAIEmbeddings, index: FAISS, language: str
) -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """

    query_vector = embeddings.embed_query(question)
    context = "\n".join(
        [
            doc.page_content
            for doc in index.similarity_search_by_vector(query_vector, k=2)
        ]
    )
    prompt = cp.ANSWER_GEN_PROMPT.format(question=question, context=context)

    prompts = [
        SystemMessage(content=cp.ANSWER_GEN_SYSTEM_PROMPT),
        HumanMessage(content=cp.ANSWER_GEN_QUESTION_EXAMPLE_1),
        AIMessage(content=cp.ANSWER_GEN_ANSWER_EXAMPLE_1),
        HumanMessage(content=prompt),
    ]
    summary_prompts = [
        SystemMessage(content=cp.SUMMARY_SYSTEM_PROMPT.format(language=language)),
        HumanMessage(
            content=cp.SUMMARY_HUMAN_PROMPT.format(language=language, content=context)
        ),
    ]
    return prompts, context, summary_prompts


def evol_instruct(questions: List[str], client: AzureOpenAILLM) -> List[str]:
    """Mutates `x` questions in term of complexity following WizardLM approach and add these mutates questions to the original list

    Args:
        questions (List[str]): _description_
        client (AzureOpenAILLM): _description_

    Returns:
        List[str]: _description_
    """

    evol_instruct = EvolInstruct(
        name="evol-instruct",
        num_evolutions=2,
        input_batch_size=8,
        llm=client,
        pipeline=Pipeline(name="evol-instruct-pipeline"),
        # store_evolutions=True,
        # include_original_instruction=True,
        mutation_templates=cp.MUTATION_TEMPLATES,
    )
    evol_instruct.load()
    queries_to_mutate = []
    for q in questions:
        queries_to_mutate.append({"instruction": q})

    result = next(evol_instruct.process(queries_to_mutate))

    evolved_questions = [ev["evolved_instruction"] for ev in result]
    evolved_questions.extend(questions)

    return evolved_questions


def add_chunk_to_dataset(
    chunk: dict,
    client: AzureChatOpenAI,
    client_3_5: AzureChatOpenAI,
    mutation_client: AzureOpenAILLM,
    number_of_questions: int,
    splitter: CharacterTextSplitter,
    embeddings: AzureOpenAIEmbeddings,
) -> list[dict]:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """

    q_list = []

    questions = generate_instructions_gen(
        chunk=chunk, number_of_questions=number_of_questions, client=client
    )
    if not questions or questions[0] == "<NOQUESTION>":
        return [{"status": "<NOQUESTION>"}]

    if random.random() < 0.5:
        questions = evol_instruct(questions=questions, client=mutation_client)

    with open(
        chunk["full_document_path"],
        "r",
    ) as f:
        full_doc = json.load(f)
        full_content = full_doc["content"]
    docs = splitter.split_text(full_content)
    index = init_index(
        embeddings, docs
    )  # 1.30 minuti per le domande, indexing 10 secondi, 30 secondi per la generazione delle risposte con gpt-3-5
    compiled_prompts, contexts, summary_prompts = list(
        map(
            list,
            zip(
                *[
                    encode_question_gen(
                        question=question,
                        embeddings=embeddings,
                        index=index,
                        language=map_language_flags(chunk["language"]),
                    )
                    for question in questions
                ]
            ),
        )
    )
    try:
        answers = client_3_5.batch(
            inputs=compiled_prompts, config={"max_concurrency": 3}
        )
        sumamries = client_3_5.batch(
            inputs=summary_prompts, config={"max_concurrency": 3}
        )
    except BadRequestError:
        return [{"status": "too many tokens"}]
    answers_contents = [answer.content for answer in answers]
    summary_contents = [summary.content for summary in sumamries]
    for question, answer, context, summary in zip(
        questions, answers_contents, contexts, summary_contents
    ):
        datapt = {
            "id": hash_value(chunk),
            "question": question,
            "answer": answer,
            "context": context,
            "summary": summary,
        }

        q_list.append(
            {
                **datapt,
                **{
                    key: value
                    for key, value in zip(chunk.keys(), chunk.values())
                    if key not in ["chunk"]
                },
            }
        )
    return q_list


def map_language_flags(flag: str) -> str:
    mapping = {"en": "english", "it": "italian"}
    return mapping[flag]


def init_index(embeddings: AzureOpenAIEmbeddings, documents: List[str]):

    index = faiss.FAISS.from_texts(documents, embeddings)
    return index
