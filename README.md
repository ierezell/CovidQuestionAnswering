## Botpress Question Answering

This module goal is to allow botpress to reply questions from textual documents in an unsupervised way.

At the end we should be able to give it a website Url, pdfs or differents documents and it will act like a custom search engine

It is written in python and will run in a docker which will then be integrated with a module calling the api

## Assumptions

- We assume the query is a short question
- We assume only one language
- We assume the query is in the scope of the documents (COVID-related)
- We assume there's always a relevant document for the query

## Milestones


- [x] Full pipeline along with interfaces between components
- [ ] Retriever yeilds decent results when querying manually
- [ ] Build retriever dataset & measure retriever performances
- [ ] TBD
