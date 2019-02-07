import gensim
import pandas as pd
import pickle

import sys
sys.path.insert(0, './../preprocessing')
import session as ss

def model_to_CSV(model, embedding_size):
    labels = []
    tokens = []
    print('vocabulary length',len(model.wv.vocab))

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    tokens_df = pd.DataFrame(tokens)
    labels_df = pd.DataFrame(labels)
    #tokens_df.to_csv('word2vec_embeddings/embedding_%d.csv' % embedding_size,sep='\t', index=False, header=None)
    #labels_df.to_csv('word2vec_embeddings/labels_%d.csv' % embedding_size,sep='\t', index=False, header=None)
    tokens_df.to_csv('word2vec_preprocessed_embeddings/embedding_%d.csv' % embedding_size,sep='\t', index=False, header=None)
    labels_df.to_csv('word2vec_preprocessed_embeddings/labels_%d.csv' % embedding_size,sep='\t', index=False, header=None)

df = pickle.load(open("./../data_set.p", "rb"))

#cleaned data
df = df[(df['action_cleaned']!='click_on_other')]

# create session UUID
d = ss.define_session(df)

corpus = []
for uuid, row in d.groupby('UUID'):
    session = []
    for a in row['action_cleaned']:
        session.append(a)
    corpus.append(session)

total_examples = len(corpus)
print('num sessions', total_examples)
print('total num events',len(d))    

embedding_size = 110

model = gensim.models.Word2Vec(corpus, size=embedding_size, window=3, min_count=2, workers=10)

model.train(corpus,total_examples=total_examples,epochs=30)

model_to_CSV(model,embedding_size)




