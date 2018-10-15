# !/usr/bin/env python3
# coding: utf-8

## Install necessary dependencies.
##
## scpacy (https://spacy.io/usage/#section-quickstart):
##     $ pip install -U spacy
## if you use conda:
##     $ conda install -c conda-forge spacy
## spacy's English language model:
##     $ python -m spacy download en
## gensim (https://radimrehurek.com/gensim/install.html):
##     $ pip install --upgrade gensim
## or, if you use conda:
##     $ conda install -c conda-forge gensim
# In[22]:


import re
import spacy
from gensim import corpora # necessary for building dictionary and bag of words representation
from gensim import models # necessary for Phrases, LDA model and TFIDF

## If there's time, could also work out how to visualise the results with PyLDA
# In[23]:


nlp = spacy.load("en")


# In[24]:


corpus = "snf_html_corpus_simple.txt"

raw_corpus = []

with open(corpus, "r", encoding="utf8") as inf:
    doc = ""
    for line in inf:
        if not line.startswith("\n"): # we are inside a document
            line = re.sub("\xa0", " ", line.rstrip())
            doc += line.lower() # lower and add the line to the document string
        else: # we are at a document boundary
            if len(doc) > 380: # ignore shorter documents
                raw_corpus.append(doc) # should end up with 110 articles in list of docs
            doc = ""

# inspect the raw_corpus:
# raw_corpus


# In[25]:


# testing on one doc:

# pro_doc = nlp(docs[0])
# print(pro_doc)

# Inspect entities found
# pro_doc.ents

# for token in pro_doc[:100]:
#     print(token.text, token.lemma_, token.pos_, token.tag_,
#           token.dep_, token.shape_, token.is_alpha, token.is_stop)


# In[26]:


# add stopwords
noisy_words = ["research", "say", "says", "said", "in", "not", "this", "researcher"]

for w in noisy_words:
    nlp.vocab[w].is_stop = True


# In[27]:


# test = nlp("The research says that this man is not a researcher, but he is.".lower())
# for w in test:
#     if not w.is_stop:
#         print(w)


# In[28]:


corpus, article = [], []

for doc in raw_corpus:
    doc = nlp(doc)
    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article!
        if w.text != " " and not w.is_stop and not w.is_punct and not w.like_num:
            # we add the lematized version of the word
            lemma = w.lemma_.strip("’") # removes things like "science’"
            if lemma:
                article.append(lemma)
    corpus.append(article)
    article = []

# inspect corpus
corpus

## Using Gensim's phrases model, we can easily identify some MWEs in the corpus.
## This puts words like 'open' and 'science' into a bigram 'open_science'.
# In[29]:


trained_phrases = models.phrases.Phrases(corpus) # train a phrase model
phraser = models.phrases.Phraser(trained_phrases) # initialise the trained phraser model
corpus_phrased = [phraser[text] for text in corpus] # apply phraser model to each text in corpus
# inspect the first text in the corpus after applying the phrase model
# corpus_phrased[3]


# In[30]:


dictionary = corpora.Dictionary(corpus_phrased)
mapped_corpus = [dictionary.doc2bow(doc) for doc in corpus_phrased]
# inspect mapped_corpus
# corpus[1][0:10]


# In[31]:


def print_topics_gensim(topic_model, total_topics=1, weight_threshold=0.0001, num_terms=None):
    """Adapted pretty print topic model results from Sakar (2016) Text Analytics With Python."""
    for index in range(total_topics):
        topic = topic_model.show_topic(index)
        topic = [(word, round(wt,4)) for word, wt in topic if abs(wt) >= weight_threshold]
        print('Topic #'+str(index+1))
        print(topic[:num_terms] if num_terms else topic)
        print()

def train_lda_model_gensim(corpus, total_topics=2):
    """Adapted pretty print topic model results from Sakar (2016) Text Analytics With Python."""
    dictionary = corpora.Dictionary(corpus)
    mapped_corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, iterations=1000, num_topics=total_topics)
    return lda


# In[32]:


def run_model_for_X_topics(X=2):
    lda_gensim = train_lda_model_gensim(corpus_phrased, total_topics=X)
    print_topics_gensim(topic_model=lda_gensim, total_topics=X, num_terms=10)


# In[33]:


run_model_for_X_topics(10)
