import logging
import re
from datetime import datetime
from typing import Any, List, Tuple

from datatypes import Answer, Chunk, Models
from embedders.answerer import Answerer
from utils import cosine_similarity, get_keylemmas

logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)


def get_best_span(chunk: Chunk, question: str, qa_model: Answerer, nlp: Any
                  ) -> Tuple[str, float, List[str]]:
    """Get the best full sentence answering the question within a context.
    If possible add the next sentence as well

    Args:
        context (str): Paragraph in which we want a sentence as answer
        question (str): A one line short question (should be interogative)
        qa_model (Answerer): The deep learning model
        nlp (Any): Spacy preprocessor for lemmatization, sentecizer, tokenizer
    """
    qa_ans = qa_model.answer(question, chunk)
    match = re.match(r'(^.+)\.', qa_ans['answer'])
    chunk_sents: List[str] = [s.text for s in nlp(chunk['content']).sents]
    clean_ans = match.groups()[0] if match else qa_ans['answer']

    # Get the full sentence from the elected span
    ans_sent, idx = [(s, i) for i, s in enumerate(chunk_sents)
                     if clean_ans in s][0]

    # If possible add the sentence after the elected span as well
    if idx + 1 < len(chunk_sents):
        ans_sent += chunk_sents[idx+1]

    links = []
    for link in chunk['links']:
        if get_keylemmas(link['name'], nlp).intersection(
                get_keylemmas(question, nlp)):
            links.append(link['path'])
    return ans_sent, qa_ans['score'], links


def answer_question_by_chunks(question_embed: List[float], question: str,
                              supports: List[Chunk], models: Models) -> Answer:
    """
    Elect the best document among the supports.
    - First elect a chunk based on deep QA, keywords, and cosine
    - Then if span election is too low, get a span from keywords
    """

    best_ans: Answer = {"content": '', "title": '', "score": 0.0,
                        "start": 0, "end": 0, 'elected': 'n/a',
                        'date': datetime.now(), 'link': [''],
                        'answer': "No answer found, try to reformulate"}
    if not supports:
        return best_ans

    best_score: float = 0.0
    best_chunk: Chunk = None  # type: ignore
    nlp = models['processor']['fr']

    lem_question = get_keylemmas(question, nlp)

    # Docs one by one
    for chunk in supports:
        ans_sent, ans_score, links = get_best_span(
            chunk=chunk, question=question, nlp=nlp,
            qa_model=models['answerer']['fr'])

        ans_embed = models['embedder']['fr'].embed(ans_sent, "sentence")
        score_embed = cosine_similarity(question_embed, ans_embed)

        lem_ans = get_keylemmas(ans_sent, nlp)
        lem_title = get_keylemmas(chunk['title'], nlp)
        lem_content = get_keylemmas(chunk['content'], nlp)

        score_keywords = (len(lem_question.intersection(lem_ans)) +
                          len(lem_question.intersection(lem_content))
                          + len(lem_question.intersection(lem_title))
                          ) / len(lem_question)

        score = score_keywords + score_embed + ans_score

        if score > best_score:
            best_ans = {'score':  ans_score, 'answer': ans_sent,
                        'content': chunk['content'],
                        "title": chunk['title'],
                        'date': chunk['first_seen_date'],
                        'start': chunk['content'].index(ans_sent),
                        'end': chunk['content'].index(ans_sent)+len(ans_sent),
                        'elected': 'qa',
                        'link': links}
            best_chunk = chunk
            best_score = score

    if best_ans['score'] < 0.5:
        best_ans: Answer = {"content": '', "title": '', "score": 0.0,
                            "start": 0, "end": 0, 'elected': 'n/a',
                            'date': datetime.now(), 'link': [''],
                            'answer': "No answer found, try to reformulate"}

        for sent in [s.text for s in nlp(best_chunk['content']).sents]:
            sent_score = cosine_similarity(
                models['embedder']['fr'].embed(sent, "sentence"),
                question_embed)

            if sent_score > best_ans['score']:
                start = best_chunk['content'].index(sent)
                best_ans = {'score': sent_score, 'answer': sent,
                            'content': best_chunk['content'],
                            'title': best_chunk['title'],
                            'date': best_chunk['first_seen_date'],
                            'start': start, 'end': start+len(sent),
                            'elected': 'kw',
                            'link': [
                                li['path'] for li in best_chunk['links']
                                if get_keylemmas(li['name'], nlp).intersection(
                                    lem_question)
                            ]}

    return best_ans
    # # Concat all docs and find answer
    # all_text = []
    # for chunk in supports:
    #     all_text.append(chunk['content'])
    # return models['answerer']['fr'].answer(question, all_text)


def answer_question_by_summary(supports: List[Chunk], models: Models
                               ) -> Answer:
    summary = models['summarizer']['fr'].summarize(supports)
    return summary
