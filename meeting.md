# Tests :

https://www.quebec.ca/sante/problemes-de-sante/a-z/coronavirus-2019/reponses-questions-coronavirus-covid19/isolement-symptomes-traitements-covid-19/#c54105

- Existe il un vaccin pour le covid-19 // bad QA but doc in the 10
- Existe-t-il un traitement contre la COVID‑19 ?
- Est-ce qu'il existe un vaccin contre le COVID-19 // Good answer

- Multi chunk appears many time (because of title)

# Done :

- Computing on GPU 200s vs 600 for cpu
- Boost by date (with decay)

# Doing :

- Better filter / Election (Ex : question "Plop" returns documents)
- Test embdding all vs sentence
- add other field boosting
- Add NER/lemmatization to retrieve voc with same semantic structure

# Ideas :

- Move the clear_database button further from the ask button (stupid missclick)
- Baseline done : Need a dataset to test the solutions

# TLDR :

- Question working good if well formulated, the rest is still returning crap -> need better cleaning and election
