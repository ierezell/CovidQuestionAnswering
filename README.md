## Botpress Question Answering

This module goal is to allow botpress to reply questions from textual documents in an unsupervised way.

At the end we should be able to give it a website Url, pdfs or differents documents and it will act like a custom search engine

It is written in python and will run in a docker which will then be integrated with a botpress module calling the api

## Code organization :

- Precomputed datasets are in the dataset folder.
- The embedder folder is a wrapper for the deeplearning models (embedding and QA)
- Indexer folder is responsible for all the preprocessing
- Qa folder is responsible for all the retrieval / inference
- `config.py` stores all the useful global variables like model names
- `datatypes.py` stores all datatypes used in the fonctions for type hints
- `utils.py` provides some global standalone functions like sanitazing/hashing text or math like a cosine similarity
- `test.py` is made for developpement only to be sure that the code runs without trying all the interactive streamlit things.

## Installation & Running :

- Make sure you have python at least 3.8
- Optional but prefered : Make a virtual environement
- Install all the dependecies with `pip install -r requirements.txt`
- Run the code with `streamlit run pipeline.py`. It will open a tab in your default browser
- First time running, the database will be computed when you click on the `ask` button, it takes time (more than 5mn on cpu). Subsequent question will use the same database so it will be fast.

N.b : Eqch times when asking the first question (after database is created) all the models needs to be loaded so expect ~50s of overhead. Then subsequent questions are fast.

## Method :

- Preprocess

  - Clean the html data (already parsed in a tree manner with children but maybe will do a parser with scrapy)
  - Chunk the documents in pieces
  - Compute useful metadatas
  - Index this chunks with the metadatas in a database

- Retrieving

  - Query the database with infos from botpress (like topics) to retrieve the X more pertinents docs
  - Among those docs elect the best sentence to answer the question

## Assumptions

- We assume the query is a short question
- We assume only one language (french for now)
- We assume the query is in the scope of the documents (COVID-related)
- We assume there's always a relevant document for the query

## Milestones

- [x] Full pipeline along with interfaces between components
- [ ] Retriever yields decent results when querying manually
- [ ] Build retriever dataset & measure retriever performances
- [ ] TBD
