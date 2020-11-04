# Tests :

https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/reponses-questions-coronavirus-covid19/isolement-symptomes-traitements-covid-19/#c54105

- Existe il un vaccin pour le covid-19 // bad QA but doc in the 10
- Existe-t-il un traitement contre la COVID‑19 ?
- Est-ce qu'il existe un vaccin contre le COVID-19 // Good answer

- Combien de deces -> No good answer

- Comment soulager les maux de gorge?
- Comment faire pour atténuer le mal à la gorge? Give two different results
  both correct, one hitting "maux" the other "mal" even with embeddings
  can be corrected with lem

- Multi chunk appears many time (because of title)

# Done :

- Computing on GPU 200s vs 600 for cpu
- Boost by date (with decay)
- Better refinder

# Doing :

- Better filter / Election (Ex : question "Plop" returns documents)
- Test embdding all vs sentence
- add other field boosting
- Add NER/lemmatization to retrieve voc with same semantic structure (spacy or es builtin)
- Add content of last parent with content

# Ideas :

- Baseline done : Need a dataset to test the solutions
- Test differents embedders with dataset
- Lemmatize text and embed it with fasttext because no more context
- Summarize chunks with BART

# TLDR :

- Question working good if well formulated, the rest is still returning crap -> need better cleaning and election

# Done but to check

- Embeddings : Embed full paragraph vs embed all sentences and mean them
- Cosine : weights of all types
- Chunk size (be careful not too big for the embedder)
