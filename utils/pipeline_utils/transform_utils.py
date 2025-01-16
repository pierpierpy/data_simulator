from typing import List
from langchain_core.documents.base import Document
import random
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import src.utils.custom_prompts as cp
from transformers import Pipeline
import numpy as np


def democratic_language_tagger(text: List[List[str]], pipeline: Pipeline, **kwargs) -> str:

    result = {}

    for page in text:
            page_languages = pipeline(page, top_k = 1, truncation = True)
            for language in page_languages:
                
                if language[0]["label"] not in result.keys():
                    result[language[0]["label"]] = language[0]["score"]
                result[language[0]["label"]] += language[0]["score"]
    probs = softmax(result)
    d_language = max(
        probs, key=result.get
    )
    return d_language, probs[d_language] 


def sample_docs(text: List[str], n_phrases: int = 4, **kwargs) -> List[List[str]]:
    doc_samples = random.choices(text, **kwargs)
    return [
        [phrase for phrase in doc_sample.split(".")][0:n_phrases]
        for doc_sample in doc_samples
    ]

def get_context(texts: List[str], client:AzureChatOpenAI ):
    text = "\n\n".join(texts)
    return client.invoke([SystemMessage(content= cp.CONTEXT_SYSTEM_PROMPT), HumanMessage(content = cp.CONTEXT_HUMAN_PROMPT.format(text = text))])


def softmax(dictionary):
    values = np.array(list(dictionary.values()))
    exp_values = np.exp(values - np.max(values))  # Subtracting max value for numerical stability
    softmax_values = exp_values / np.sum(exp_values)
    
    softmax_dict = {}
    keys = list(dictionary.keys())
    for i in range(len(keys)):
        softmax_dict[keys[i]] = softmax_values[i]
    
    return softmax_dict


def democratic_language_tagger_detect_lang(text: List[List[str]], **kwargs) -> str:

    result = {}

    for page in text:
        for phrase in page:
            try:
                language = detect_langs(phrase)[0]
            except LangDetectException:
                continue  # TODO skipping the errors is not ok
            if language.lang not in result.keys():
                result[language.lang] = language.prob
            result[language.lang] += language.prob
    probs = softmax(result)
    d_language = max(
        probs, key=result.get
    )
    return d_language, probs[d_language] 