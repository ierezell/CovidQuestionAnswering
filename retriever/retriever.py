from datatypes import Indexes, Models
from typing import Any, List, Tuple
import numpy as np


def retrieve_docs(indexes: Indexes, models: Models, question: str
                  ) -> Tuple[List[float], List[Any], float, int]:
    question_embed = models['embedder']['multi'].embed(question)
    supports, max_score, hits = retrieve_es(indexes['db'],
                                            question, question_embed)
    return question_embed, supports, max_score, hits


# def retrieve_faiss(faiss: Any, faiss_idx: List[str],
#                    question: str, question_embed: List[float]):
#     distances, indices = faiss.search(np.array([question_embed],
#                                                dtype=np.float32),
#                                       5)

#     return supports, max_score, hits


def retrieve_es(es: Any, question: str, question_embed: List[float]):
    es_query_body = {
        "query": {
            "script_score": {
                "query": {
                    'multi_match': {'query': question,
                                    'fuzziness': "AUTO",
                                    'fields': ['title', 'content']
                                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.question_embed, doc['content_embedding']) + 1.0",  # noqa E501
                    "params": {"question_embed": question_embed}
                }
            }
        }
    }

    res = es.search(index="gouv", body=es_query_body)
    max_score = res['hits']['max_score']
    hits = res['hits']['total']
    print("HITS  ", hits)

    supports = []
    for doc in res['hits']['hits']:
        supports.append({'score': doc['_score'], **doc["_source"]})

    return supports, max_score, hits
