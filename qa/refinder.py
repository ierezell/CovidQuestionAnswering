from datatypes import Models
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from utils import cosine_similarity
from typing import List, Set
import logging
import re

logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)


def get_keywords(text: str, processor) -> Set[str]:
    return {w.text.lower() for w in processor(text)
            if w.text.lower() not in fr_stop}


def answer_question(question_embed: List[float], question: str,
                    supports, models: Models):
    if not supports:
        return {"content": '', 'answer': "No answer found, try to reformulate",
                "score": 0, "start": 0, "end": 0, 'elected': 'N/a1'}
    best_ans = None
    best_score = 0
    best_chunk = None
    nlp = models['processor']['fr']

    # Docs one by one
    keywords_question = get_keywords(question, nlp)

    for chunk in supports:
        chunk['content'] = chunk['content'].replace('_', '')
        chunk_sents = [s.text for s in nlp(chunk['content']).sents]

        ans = models['answerer']['fr'].answer(question, chunk['content'])
        match = re.match(r'(^.+)\.', ans['answer'])
        clean_ans = match.groups()[0] if match else ans['answer']
        # print(clean_ans)
        # print(chunk_sents)
        ans_sent = [s for s in chunk_sents if clean_ans in s][0]
        ans_embed = models['embedder']['fr'].embed(ans_sent)
        score_embed = cosine_similarity(question_embed, ans_embed)

        keywords_ans = get_keywords(ans['answer'], nlp)
        keywords_title = get_keywords(chunk['title'], nlp)
        keywords_content = get_keywords(chunk['content'], nlp)

        score_keywords = (len(keywords_question.intersection(keywords_ans)) +
                          len(keywords_question.intersection(keywords_content))
                          + len(keywords_question.intersection(keywords_title))
                          ) / len(keywords_question)

        score = score_keywords + score_embed + ans['score']

        # score = score_embed
        if score > best_score:
            best_ans = {'score':  ans['score'], 'answer': ans_sent,
                        'start': chunk['content'].index(ans_sent),
                        'end': chunk['content'].index(ans_sent)+len(ans_sent),
                        'elected': 'qa'}
            best_chunk = chunk
            best_score = score

    if best_ans['score'] < 0.2:
        best_ans = {'score': 0, 'answer': '', 'start': 0, 'end': 0}
        for sent in [s.text for s in nlp(best_chunk['content']).sents]:
            sent_score = cosine_similarity(
                models['embedder']['fr'].embed(sent), question_embed)

            if sent_score > best_ans['score']:
                start = best_chunk['content'].index(sent)
                best_ans = {'score': sent_score, 'answer': sent,
                            'start': start, 'end': start+len(sent),
                            'elected': 'kw'}

    return dict({'content': best_chunk['content']}, **best_ans)
    # # Concat all docs and find answer
    # all_text = []
    # for chunk in supports:
    #     all_text.append(chunk['content'])
    # return models['answerer']['fr'].answer(question, all_text)
