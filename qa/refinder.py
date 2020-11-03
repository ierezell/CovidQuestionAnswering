import logging
import re
from typing import List, Set

from datatypes import Answer, Chunk, Models
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from utils import cosine_similarity

logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)


def get_keywords(text: str, processor) -> Set[str]:
    return {w.text.lower() for w in processor(text)
            if w.text.lower() not in fr_stop}


def answer_question(question_embed: List[float], question: str,
                    supports: List[Chunk], models: Models) -> Answer:

    best_ans: Answer = {"content": '', "score": 0.0, "start": 0, "end": 0,
                        'answer': "No answer found, try to reformulate",
                        'elected': 'n/a'}
    if not supports:
        return best_ans

    best_score: float = 0.0
    best_chunk: Chunk
    nlp = models['processor']['fr']

    # Docs one by one
    keywords_question = get_keywords(question, nlp)

    for chunk in supports:
        chunk['content'] = chunk['content'].replace('_', '')
        chunk_sents: List[str] = [s.text for s in nlp(chunk['content']).sents]

        ans = models['answerer']['fr'].answer(question,
                                              chunk['content'])
        match = re.match(r'(^.+)\.', ans['answer'])
        clean_ans = match.groups()[0] if match else ans['answer']

        ans_sent: str = [s for s in chunk_sents if clean_ans in s][0]
        ans_embed = models['embedder']['fr'].embed(ans_sent, "sentence")
        score_embed = cosine_similarity(question_embed, ans_embed)

        keywords_ans = get_keywords(ans['answer'], nlp)
        keywords_title = get_keywords(chunk['title'], nlp)
        keywords_content = get_keywords(chunk['content'], nlp)

        score_keywords = (len(keywords_question.intersection(keywords_ans)) +
                          len(keywords_question.intersection(keywords_content))
                          + len(keywords_question.intersection(keywords_title))
                          ) / len(keywords_question)

        score = score_keywords + score_embed + ans['score']

        if score > best_score:
            best_ans = {'score':  ans['score'], 'answer': ans_sent,
                        'content': chunk['content'],
                        'start': chunk['content'].index(ans_sent),
                        'end': chunk['content'].index(ans_sent)+len(ans_sent),
                        'elected': 'qa'}
            best_chunk = chunk
            best_score = score

    if best_ans['score'] < 0.2:
        best_ans: Answer = {"content": '', "score": 0.0, "start": 0, "end": 0,
                            'answer': "No answer found, try to reformulate",
                            'elected': 'n/a'}
        for sent in [s.text for s in nlp(best_chunk['content']).sents]:
            sent_score = cosine_similarity(
                models['embedder']['fr'].embed(sent, "sentence"),
                question_embed)

            if sent_score > best_ans['score']:
                start = best_chunk['content'].index(sent)
                best_ans = {'score': sent_score, 'answer': sent,
                            'content': best_chunk['content'],
                            'start': start, 'end': start+len(sent),
                            'elected': 'kw'}

    return best_ans
    # # Concat all docs and find answer
    # all_text = []
    # for chunk in supports:
    #     all_text.append(chunk['content'])
    # return models['answerer']['fr'].answer(question, all_text)
