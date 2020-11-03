"""
This file is made to run the streamlit application.

You need to call it with streamlit run pipeline.py
"""
import json
import logging
from typing import Tuple, List, Any

import elasticsearch
import streamlit as st

from datatypes import Indexes, RawEntry, Models
from indexer.indexer import preprocess
from qa.refinder import answer_question
from qa.retriever import retrieve_docs

st.title('Gouv bot')
logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)


# @st.cache(hash_funcs={elasticsearch.Elasticsearch: id,
#                       "preshed.maps.PreshMap": id,
#                       "cymem.cymem.Pool": id,
#                       "thinc.model.Model": id,
#                       "spacy.pipeline.tok2vec.Tok2VecListener": id})
def ask_question(indexes: Indexes, models: Models, question: str
                 ) -> Tuple[List[float], List[Any], float, int]:
    """Query the database to retrieve the 10 most pertinent documents

    Args:
        indexes (Indexes)
        models (Models)
        question (str) : The user question like "What are the symptoms"

    Returns:

    """
    return retrieve_docs(indexes, models, question)


@st.cache(allow_output_mutation=True)
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
    indexes, models = preprocess(data)

    return indexes, models


def clear_indexes(indexes: Indexes):
    """Remove all the indexes (elasticsearch) to start from scratch

    Args:
        indexes (Indexes): Indexes (for now just db : elasticsearch)
    """
    for index in indexes['db'].indices.get('*'):
        indexes['db'].indices.delete(index=index, ignore=[400, 404])


indexes = None  # type: ignore
user_input = st.text_input("Posez une question",
                           "Les femmes enceintes sont-elles plus a risque ?")

if st.button('ask'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.success('Done!')
    question_embed, supports, max_score, hits = ask_question(indexes, models,
                                                             user_input)

    with st.beta_expander("See the supports documents"):
        st.write("Max Score")
        st.write(max_score)
        st.write("Hits")
        st.write(hits)
        st.write("Docs")
        st.write([
            {
                k: v for k, v in sup.items()
                if k in ['title', 'content', 'parent_title',
                         'path', 'first_seen_date']
            }
            for sup in supports
        ])

    answer = answer_question(question_embed, user_input, supports, models)

    st.write(answer)

if st.button('clear database'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.write("Clearing")
    clear_indexes(indexes)
    st.success('Done!')
