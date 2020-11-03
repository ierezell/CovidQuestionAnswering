"""
This file is made to run the streamlit application.

You need to call it with streamlit run pipeline.py
"""
import json
import logging
from typing import Tuple, List, Any

from datatypes import Indexes, RawEntry, Models
from indexer.indexer import preprocess
from qa.refinder import answer_question
from qa.retriever import retrieve_docs

logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)

embedding_mode = "all"
TEST_THEN_CLEAR = False


def ask_question(indexes: Indexes, models: Models, question: str
                 ) -> Tuple[List[float], List[Any], float, int]:
    """Query the database to retrieve the 10 most pertinent documents

    Args:
        indexes (Indexes)
        models (Models)
        question (str) : The user question like "What are the symptoms"

    Returns:

    """

    return retrieve_docs(
        f"gouv_{embedding_mode}", indexes, models, question,
        options={
            'retrieve_nb': 10,
            'boost_lem': 1.0,
            'boost_ner': 1.0,
            'boost_date': 1.0,
            'boost_title': 1.0,
            'boost_content': 1.0,
            'boost_parent_title': 1.0,
            'boost_parent_content': 1.0,
            'boost_title_embedding': 1.0,
            'boost_parent_embedding': 1.0,
            'boost_content_embedding': 1.0,
            'embedding_mode': embedding_mode,
        })


def preprocess_data() -> Tuple[Indexes, Models]:
    """Fill the indexes and create the models to interact with it

    Returns:
        Tuple[Indexes, Models]:
            Indexes : for now just elasticsearch with all chunks processed
            Models : Embedder, Q&A and spacy processor
    """
    with open('datasets/gouv/DOCUMENTS.json', 'r') as file:
        # with open('datasets/gouv/dummy.json', 'r') as file:
        data: RawEntry = json.load(file)
    indexes, models = preprocess(f"gouv_{embedding_mode}", data)

    return indexes, models


def clear_indexes(indexes: Indexes):
    """Remove all the indexes (elasticsearch) to start from scratch

    Args:
        indexes (Indexes): Indexes (for now just db : elasticsearch)
    """
    for index in indexes['db'].indices.get('*'):
        indexes['db'].indices.delete(index=index, ignore=[400, 404])


indexes = None  # type: ignore
user_input = "Les femmes enceintes sont-elles plus a risque ?"
indexes, models = preprocess_data()
question_embed, supports, max_score, hits = ask_question(indexes,
                                                         models,
                                                         user_input)
answer = answer_question(question_embed, user_input, supports, models)

if TEST_THEN_CLEAR:
    clear_indexes(indexes)
