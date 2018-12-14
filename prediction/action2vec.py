import gensim
import pandas as pd

import sys
sys.path.insert(0, './../preprocessing')
import session as ss

def model_to_CSV(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tokens_df = pd.DataFrame(tokens)
    labels_df = pd.DataFrame(labels)
    tokens_df.to_csv('tokens.csv',sep='\t', index=False, header=None)
    labels_df.to_csv('labels.csv',sep='\t', index=False, header=None)

t = ss.define_session()

corpus = []
for uuid, row in t.groupby('UUID'):
    session = []
    for a in row['action']:
        session.append(a)
    corpus.append(session)

model = gensim.models.Word2Vec(corpus, size=150, window=5, min_count=2, workers=10)
model.train(corpus,total_examples=len(corpus),epochs=10)

model_to_CSV(model)




