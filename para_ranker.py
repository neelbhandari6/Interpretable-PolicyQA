# rom gensim.test.utils import datapath, get_tmpfile

# from gensim.models import KeyedVectors

# from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import numpy as np
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# glove_file = 'glove.6B.100d.txt'

# tmp_file = get_tmpfile("test_word2vec.txt")


# _ = glove2word2vec(glove_file, tmp_file)


# model = KeyedVectors.load_word2vec_format(tmp_file)


nlp = spacy.load('en_core_web_sm')

def get_ranks(paras,question):
    full_corpus_tokens=[]
    for x in paras:
        doc = nlp(x)
        tokens=[token.text.lower() for token in doc if token.is_stop == False and token.text.isalpha() == True]

        full_corpus_tokens.append(tokens)
    qdoc = nlp(question)
    qtokens=[token.text.lower() for token in qdoc if token.is_stop == False and token.text.isalpha() == True]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(full_corpus_tokens)]

    dmodel = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4) 
    qvec=dmodel.infer_vector(qtokens)
    scores=[]
    for i in range(len(full_corpus_tokens)):
        ele=dmodel.docvecs[i]
        scores.append(np.dot(ele,qvec)) 
    
    return gensim.matutils.argsort(scores)[::-1]
    
    