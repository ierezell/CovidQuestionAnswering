from typing import Any, List, Tuple, cast

from datatypes import Indexes, Answer, Models


def retrieve_docs(indexes: Indexes, models: Models, question: str
                  ) -> Tuple[List[float], List[Any], float, int]:

    question_embed = models['embedder']['fr'].embed(question, "all")

    supports, max_score, hits = retrieve_es(indexes['db'], question,
                                            question_embed)

    return question_embed, supports, max_score, hits


def retrieve_es(es: Any, question: str, question_embed: List[float]):
    es_query_body = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {},  # For cosine only
                    # 'multi_match': {'query':  question,
                    #                 'fuzziness': "AUTO",
                    #                 "type": "most_fields",
                    #                 # cross_fields, most_fields, best_fields
                    #                 'fields': [
                    #                     'title',
                    #                     'content',
                    #                     'parent_title^0.5',
                    #                     'parent_content^0.5'
                    #                     ]
                    #                 }
                },
                "script": {
                    "source": "1.0 * cosineSimilarity(params.question_embed, 'content_embedding') + 1.0 * cosineSimilarity(params.question_embed, 'title_embedding') + 0.5 * cosineSimilarity(params.question_embed, 'parent_title_embedding') + 2.5",  # noqa E501
                    "params": {"question_embed": question_embed}
                }
            }
        }
    }
    # es_query_body = {
    #     "query": {
    #         'multi_match': {'query':  question,
    #                         'fuzziness': "AUTO",
    #                         'fields': ['title', 'content']
    #                         }
    #     }
    # }

    res = es.search(index="gouv", body=es_query_body)
    max_score = res['hits']['max_score']
    hits = res['hits']['total']

    supports: List[Answer] = []
    for doc in res['hits']['hits']:
        supports.append(cast(Answer,
                             {'score': doc['_score'], **doc["_source"]})
                        )

    return supports, max_score, hits
