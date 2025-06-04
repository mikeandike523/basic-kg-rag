# Flanking paragraphs approach

# Very simple

# we select the top vectors, and then flood/propogate go to previous and next paragraphs until the text reaches a certain token count,
# or the paragraph to flood to (each endis independently) has cosine similarity (normalized) beyond a cutoff with respect
# to the query vector.

