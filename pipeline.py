"""
This file is made to run the streamlit application.

You need to call it with streamlit run pipeline.py
"""
import json
import logging
from typing import Any, List, Tuple

import streamlit as st

from datatypes import EmbeddingMode, Indexes, Models, RawEntry, RetrieveMode
from indexer.indexer import preprocess
from qa.refinder import answer_question_by_chunks, answer_question_by_summary
from qa.retriever import retrieve_docs

st.title('Gouv bot')
logging.getLogger("transformers.tokenization_utils_base"
                  ).setLevel(logging.ERROR)

st.sidebar.subheader("Preprocesssing options")
st.sidebar.write(
    "If choosen options are not precomputed it will take ~5mn more")

embedding_mode: EmbeddingMode = st.sidebar.selectbox(
    label="Embedding mode", options=["all", "sentence"])

dataset: RetrieveMode = st.sidebar.selectbox(
    label="Dataset", options=['dummy', 'all', 'covid'])


st.sidebar.subheader("Retrieving options")

retrieve_mode: RetrieveMode = st.sidebar.selectbox(
    label="Retrieve mode", options=['dense', 'hybrid', 'sparse'])

retrieve_nb: int = st.sidebar.slider(
    label="Number of docs", min_value=1, max_value=100, value=10, step=1)

boost_date: float = st.sidebar.slider(
    label="Date decay", min_value=0.0, max_value=10.0, value=0.1, step=0.1)

boost_lem: float = st.sidebar.slider(label="Lemmatization importance",
                                     min_value=0.0, max_value=10.0, value=1.0,
                                     step=0.1)

boost_page_lem: float = st.sidebar.slider(
    label="Page Lemmatization importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

boost_page: float = st.sidebar.slider(label="Page content importance",
                                      min_value=0.0, max_value=10.0, value=1.0,
                                      step=0.1)

boost_ner: float = st.sidebar.slider(label="NER importance", min_value=0.0,
                                     max_value=10.0, value=1.0, step=0.1)

boost_title: float = st.sidebar.slider(label="Title importance", min_value=0.0,
                                       max_value=10.0, value=1.0, step=0.1)

boost_content: float = st.sidebar.slider(label="Content importance",
                                         min_value=0.0, max_value=10.0,
                                         value=1.0, step=0.1)

boost_title_embedding: float = st.sidebar.slider(
    label="Title embedding importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

boost_parent_embedding: float = st.sidebar.slider(
    label="Parent embedding importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

boost_content_embedding: float = st.sidebar.slider(
    label="Content embedding importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

boost_parent_title: float = st.sidebar.slider(
    label="Parent title importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

boost_parent_content: float = st.sidebar.slider(
    label="Parent content importance", min_value=0.0, max_value=10.0,
    value=1.0, step=0.1)

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
    return retrieve_docs(
        f"{dataset}_{embedding_mode}", indexes, models, question,
        options={
            'retrieve_nb': retrieve_nb,
            'retrieve_mode': retrieve_mode,
            'boost_lem': boost_lem,
            'boost_page_lem': boost_page_lem,
            'boost_ner': boost_ner,
            'boost_date': boost_date,
            'boost_title': boost_title,
            'boost_content': boost_content,
            'boost_page': boost_page,
            'boost_parent_title': boost_parent_title,
            'boost_parent_content': boost_parent_content,
            'boost_title_embedding': boost_title_embedding,
            'boost_parent_embedding': boost_parent_embedding,
            'boost_content_embedding': boost_content_embedding,
            'embedding_mode': embedding_mode,
        })


@ st.cache(allow_output_mutation=True)
def preprocess_data() -> Tuple[Indexes, Models]:
    """Fill the indexes and create the models to interact with it

    Returns:
        Tuple[Indexes, Models]:
            Indexes : for now just elasticsearch with all chunks processed
            Models : Embedder, Q&A and spacy processor
    """
    with open(f'datasets/gouv/{dataset}.json', 'r') as file:
        data: RawEntry = json.load(file)
    indexes, models = preprocess(f"{dataset}_{embedding_mode}", data)

    return indexes, models


def clear_indexes(indexes: Indexes):
    """Remove the index (elasticsearch) choosen in the sidebar
    and compute a new one from scratch

    Args:
        indexes (Indexes): Indexes (for now just db : elasticsearch)
    """
    indexes['db'].indices.delete(index=f"{dataset}_{embedding_mode}",
                                 ignore=[400, 404])
    st.caching.clear_cache()


indexes = None  # type: ignore
user_input = st.text_input("Posez une question",
                           "Les femmes enceintes sont-elles plus a risque ?")

if st.button('ask'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.success('Database created sucessfully !')

    with st.spinner('Fetching...'):
        question_embed, supports, max_score, hits = ask_question(indexes,
                                                                 models,
                                                                 user_input)
    st.success(f'{retrieve_nb} docs fetched')

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

    with st.spinner('Answering...'):
        answer = answer_question_by_chunks(question_embed, user_input,
                                           supports, models)
    st.success('Answer found')
    st.write(answer)

    with st.spinner('Summarizing...'):
        answer = answer_question_by_summary(supports, models)
    st.write(answer)

if st.button('clear database'):
    with st.spinner('Processing...'):
        indexes, models = preprocess_data()
    st.write("Clearing")
    clear_indexes(indexes)
    st.success('Done!')
