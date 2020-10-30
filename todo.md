    Add stemming and language analysis in ES
    Test differents embedders with dataset
    Find a way to remove bad answers
    Add content of last parent with content
    Add content of all page (where chunk is extracted)
    Get boosting working (date + keywords)

# Done but to check

    Embeddings : Embed full paragraph vs embed all sentences and mean them
    Cosine : Cos(question, content) + Cos(question, Title)
    Split by sentence and add them to form chunk of a good size
    Chunk size (be careful not too big for the embedder)

# Type

    user query
    Doc retrieved by es
    Answer
