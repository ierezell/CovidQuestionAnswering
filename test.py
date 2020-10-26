import json
from retriever.retriever import retrieve_docs

from typing import Tuple
from datatypes import Indexes, RawEntry, Models
from indexer.indexer import preprocess
import elasticsearch

# "https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html#vector-functions"


def ask_question(indexes: Indexes, models: Models, question: str):
    return retrieve_docs(indexes, models, question)


def preprocess_data() -> Tuple[Indexes, Models]:
    # with open('datasets/gouv/DOCUMENTS.json', 'r') as file:
    with open('datasets/gouv/dummy.json', 'r') as file:
        data: RawEntry = json.load(file)
    indexes, models = preprocess(data)

    return indexes, models


def clear_elasticsearch(indexes: Indexes):
    for index in indexes['db'].indices.get('*'):
        indexes['db'].indices.delete(index=index, ignore=[400, 404])


indexes, models = preprocess_data()
user_input = input("Posez une question  :  ")
support, max_score, hits = ask_question(indexes, models, user_input)
clear_elasticsearch(indexes)
