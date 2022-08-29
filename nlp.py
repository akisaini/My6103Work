#%%
# NLP using SPACY 
import spacy
# %%
nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')
# %%
doc = nlp("Dr. Strange loves pav bhaji of mumbai as it costs only 2$ per plate. Akshat Loves wada pav though.")

for token in doc:
    print(token)
# %%
# same way to do the thing:

for sentence in doc.sents:
        for word in sentence:
            print(word)
# %%
text='''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.
gov/, and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''

# 
# %%
nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

# %%
doc = nlp(text)
# %%
links = []
for token in doc:
    if token.like_url:
        links.append(token)
        
print(links)   
# %%
import spacy
nlp = spacy.load("en_core_web_sm")
# %%
doc2 = nlp('Tesla inc. is going to acquire Twitter for $45 billion. ')
# %%
nlp.pipe_names
# %%
for ent in doc2.ents:
    print(ent.text, "|", ent.label_)
 
# %%
# importing spacy and en pipeline
import spacy
nlp = spacy.load('en_core_web_sm')
# %%
text = '''Inflation rose again in April, continuing a climb that has pushed consumers to the brink and is threatening the economic expansion, the Bureau of Labor Statistics reported Wednesday.

The consumer price index, a broad-based measure of prices for goods and services, increased 8.3% from a year ago, higher than the Dow Jones estimate for an 8.1% gain. That represented a slight ease from March’s peak but was still close to the highest level since the summer of 1982.

Removing volatile food and energy prices, so-called core CPI still rose 6.2%, against expectations for a 6% gain, clouding hopes that inflation had peaked in March.

The month-over-month gains also were higher than expectations — 0.3% on headline CPI versus the 0.2% estimate and a 0.6% increase for core, against the outlook for a 0.4% gain.

The price gains also meant that workers continued to lose ground. Real wages adjusted for inflation decreased 0.1% on the month despite a nominal increase of 0.3% in average hourly earnings. Over the past year, real earnings have dropped 2.6% even though average hourly earnings are up 5.5%.

Inflation has been the single biggest threat to a recovery that began early in the Covid pandemic and saw the economy in 2021 stage its biggest single-year growth level since 1984. Rising prices at the pump and in grocery stores have been one problem, but inflation has spread beyond those two areas into housing, auto sales and a host of other areas.

Federal Reserve officials have responded to the problem with two interest rate hikes so far this year and pledges of more until inflation comes down to the central bank’s 2% goal. However, Wednesday’s data shows that the Fed has a big job ahead.

'''
# %%
doc3 = nlp(text)
# %%
#for sentence in doc3.sents:
#    for word in sentence:
#    print(word)
#
# or 

for token in doc3:
    print(token)
# %%
nouns = []
numbers = []
for token in doc3:
    if token.pos_ == 'NOUN':
        nouns.append(token)
    elif token.pos_ == 'NUM':
        numbers.append(token)
print(nouns, numbers)    
# %%
# returns a dict
pos_count = doc3.count_by(spacy.attrs.POS)
# %%
for pos, count in pos_count.items():
    print(doc3.vocab[pos].text, '--', count, '--', spacy.explain(doc3.vocab[pos].text))
# %%
# spacy.explain gives more information about the POS - part of speech. 
import spacy
nlp = spacy.load("en_core_web_sm")
#%%
doc = nlp('Mike Bloomberg founded Bloomberg Inc. in 1982.')
# %%
for ent in doc.ents:
    print(ent.text, '|', ent.label_, '|', spacy.explain(ent.label_))
# %%
import spacy
nlp = spacy.load('en_core_web_sm')
# %%
doc = nlp('Hey what is going on with the car and Tesla inc. lately?')
# %%
for ent in doc.ents:
    print(ent.text, '|', ent.label_)
# %%
nlp.pipe_names
# %%
for token in doc:
    print(token, '|', token.pos_, '|', token.lemma_)
# %%
'Label and one hot encoding'
#Bag of words
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('spam2.csv')
# %%
df['Category'].value_counts()
#Highly unbalanced dataset. 
#ham     4825
#spam     747
# %%
df['Spam'] = df['Category'].apply(lambda x : 1 if x =='spam' else 0)

# %%
# x = message, y = result
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Spam'], test_size=0.25, random_state = 10)
# %%
X_train[:4]
y_train[:4]
# %%
from sklearn.feature_extraction.text import CountVectorizer

#creates a vocab and then generates a vector for the messages based on that vocab. 
vectorizer = CountVectorizer()
#X_train.values converts it to Numpy array. 
#type(X_train) - pandas.core.series.Series
X_train_cv = vectorizer.fit_transform(X_train.values)

# %%
X_train_cv.toarray().shape
# %%
vectorizer.get_feature_names_out()
#dir(vectorizer) will show all the methods of the vectorizer object
#vocabulary_ gives the index for the word in the vocab. 
vectorizer.vocabulary_
# %%
# Will fetch word at the 4216 index - 'me'
vectorizer.get_feature_names_out()[4216]
# %%# %%
X_train_np = X_train_cv.toarray()
# %%
#1st email is represented as this in numbers. All the words that are in the vocab are marked as 1. Rest are marked as 0. 
X_train_np[0]
# %%
np.where(X_train_np[1] != 0)
# %%
vectorizer.get_feature_names_out()[4026]
# %%
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

# %%
model.fit(X_train_cv, y_train)
# %%
X_test_cv = vectorizer.transform(X_test)

# %%
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
# %%
# STOP WORDS using SPACY

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

len(STOP_WORDS)
# %%
#'english_core_web_small' - nlp english package
nlp = spacy.load('en_core_web_sm')

# %%
doc = nlp('hey whats up')
for token in doc:
    if token.is_stop:
        print(token)
# %%
def preprocess(text):
    doc = nlp(text)
    
#token.text returns a string of the token. Just using token will return type spacy.tokens.token.Token
    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return no_stop_words

# %%
def preprocess(text):
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(no_stop_words) 
# %%
preprocess('hey whats up')
# %%
nlp.vocab['not'].is_stop = False
# %%
positive_text = preprocess('this is a good movie')
negative_text = preprocess('this is not a good movie')
# %%
def preprocess(text):
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return no_stop_words
    
text = ''' The India men's national cricket team, also known as Team India or the Men in Blue, represents India in men's international cricket.
It is governed by the Board of Control for Cricket in India (BCCI), and is a Full Member of the International Cricket Council (ICC) with Test,
One Day International (ODI) and Twenty20 International (T20I) status. Cricket was introduced to India by British sailors in the 18th century, and the 
first cricket club was established in 1792. India's national cricket team played its first Test match on 25 June 1932 at Lord's, becoming the sixth team to be
granted test cricket status.
'''
#%%
tok = preprocess(text)
# %%
len(tok)
# %%
#create a dict and iterate through token to count their value. 
frequency = {}
for i in tok:
    if i != '\n' and i != ' ':
       if i not in frequency:
           frequency[i] = 1 # if a token occurs for first time we initialize it to 1
       else:
           frequency[i] += 1 # if token is being repeated, we increment it by 1.      
            
    
# %%
[k for k, v in frequency.items() if v==max(frequency.values())]

# %%
