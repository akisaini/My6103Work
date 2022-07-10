#%%
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import gensim # word2vec
#%%
df = pd.read_csv('IMDB.csv')
# %%
df.head(10)
# %%
# Equally balanced dataset with 25000 positive and 25000 negative reviews.. 
df['sentiment'].value_counts()
# %%
# converting target variable to 1(positive) and 0. 
df['sentiment'] = df['sentiment'].apply(lambda x : 1 if x =='positive' else 0)
# %%
# running through gensim preprocessing.  - This converts each sentence into an array of words and removes unnecessary words from the sentence, converts into lowercase etc. 
review_text = df['review'].apply(lambda x : gensim.utils.simple_preprocess(x))
# %%
model = gensim.models.Word2Vec(
        # vector_size = 100 - default value
        window = 10, 
        min_count = 3, # min count of sentence
        workers = 4 # number of cpu's active
)
#%%
# building a vocabulary now. This contains a list of all the different words specific to our dataset. 
vocab = model.build_vocab(review_text, progress_per=1000)
# %%
# corpus_count is the total examples in review_text. 
model.train(review_text, total_examples=model.corpus_count, epochs = model.epochs)
# %%
model.wv.most_similar('good')
# %%
model.wv.get_index('good') # 45
#%%

'''
# key_to_index (dict) word is the key here and number is the index. 
'reinforcement': 49989, 'illuminata': 49990, 'warrick': 49991, 'northstar': 49992, 'aerobicide': 49993, 'wilted': 49994, 'raisins': 49995, 'outgrow': 49996, 'britches': 49997, 'zeb': 49998, 'arye': 49999, 'jeers': 50000, 'consign': 50001, 'scuttled': 50002, 'treacher': 50003, 'onus': 50004, 'hagerty': 50005, 'commodus': 50006, 'receipe': 50007, 'haiti': 50008, 'nibelungenlied': 50009, 'preventable': 50010, 'incontinent': 50011, 'americian': 50012, 'lamerica': 50013, 'fmlb': 50014, 'masts': 50015, 'flayed': 50016


# model.wv.index_to_key (list) list positions are the index and contain the word at the position. for example. index_to_key[45] = 'good'. 
'lightened', 'vilgot', 'strobing', 'asperger', 'darwinian', 'attrition', 'moustached', 'correlations', 'indigestible', 'yong', 'joon', 'changeable', 'sloatman', 'extremly', 'maroney', 'dispensation', 'steamers', 'excrete', 'outcrop', 'byool', 'legitimated', 'wrinkler', 'cesare', 'chipped', 'prudhomme', 'spazz'

Both are methods available to fetch total vocab size (len of either), different words in the data set and other information. Basically the vocab size is not the total length of the dataset since it has removed certain words as well.
'''
# Now we need to convert each list in review_text (which represents a review) into a list representing indexes of words.  

rev_text_main = []
for i in review_text:
    innerlist = [] # creates a new innerlist for every i
    for j in i:
        innerlist.append(model.wv.key_to_index.get(j))
    rev_text_main.append(innerlist) # append every new innerlist to main list. 

# check avgword2vec
#%%
# making sure the word is present in vocab. 
# Calculating the overall mean of dimensions of each review. Here doc represents df['review'][i] -> each seperate review.
def avgw2v(doc):
    doc =  [i for i in doc if i in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis = 0)

#%%
main = []
def rev_doc_avg(rev_doc):
    for i in rev_doc:
        main.append(avgw2v(i))
    return main
#%%
rev_doc_avg(review_text) # takes 18 mins on average to compute. 

# converting to np array
main = np.array(main)
main[0].size # 100 -> shows 100 dimensions. Each review has been converted into an np array of size (100,)
#%%
# now using train test split. 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(main, df['sentiment'], test_size=0.25, shuffle = True, random_state=15)

#%%
# building DL model. 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#%%
model = keras.Sequential([
                        keras.layers.Dense(100, input_shape= (100,), activation = 'relu'), # input layer
                        keras.layers.Dense(45, activation = 'relu'), # hidden layer (can have any number of neurons)
                        keras.layers.Dense(25, activation = 'relu'), # another hidden layer
                        keras.layers.Dense(5, activation = 'relu'), 
                        keras.layers.Dense(1, activation = 'sigmoid') # output layer -  sigmoid because target is 1/0
                        ])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20)

#%%

y_pred = model.predict(X_test)
y_p = []
for i in y_pred:
    if i[0] > 0.5:
        y_p.append(1)
    else:
        y_p.append(0)
y_p
#%%
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_p)
report = classification_report(y_test, y_p)
report # 86% accuracy


# -------------------------------------------------------------------

# %%
# word embedding. - Process of converting sentence to vector manually using padding and one hot encoding. 
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
# %%
vocab_size = 10000
# %%
# each word in review column is converted to a one_hot vector. 
# 0        [3692, 2689, 2310, 4021, 7260, 6931, 3327, 821...
# here the word 'one' is identified as a unique word with digit 1 at index 3692, it will be represented as [0,0,0,0,0,0,0,0,0,,........0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0......] total length = 10000
rev = df['review'].apply(lambda x : one_hot(x, vocab_size))
# %%
rev
# %%
rev2 = rev.tolist()
# %%
len(rev2[49999]) # 127
len(rev2[49998]) # 212
len(rev2[0]) # 314
# Now we will pad the sentences, making sure all sentences are of the same length. 
#%%
rev_len = rev.apply(lambda x : len(rev[x]))
rev_len_lt = rev_len.tolist()
rev_len_lt.sort()
# to get the largest sentence length. 
rev_len_lt[49999] # 2493
#%%
max_sentence_length = 2500

# %%
padded_revs = pad_sequences(rev, maxlen = max_sentence_length, padding = 'post')
# padded revs
padded_revs
len(padded_revs[0]) # 2500
len(padded_revs[49999]) # 2500
# %%
# Now we will describe the number of features or dimensions to develop a relation between the words. 
features = 10 # suitable number is between 10's and 100. 
model = Sequential()
model.add(Embedding(vocab_size, features, input_length = max_sentence_length, name = 'embedding'))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
# %%
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# %%
X = padded_revs
y = df['sentiment']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = True, random_state  = 15)
# %%
model.summary()
# %%
model.fit(X_train, y_train, epochs = 10, verbose = 1)
# %%
model.evaluate(X_test, y_test)

# %%
y_pred = model.predict(X_test)
# %%
y_pred_revised = []
for i in y_pred:
    if i >= 0.5:
        y_pred_revised.append(1)
    else:
        y_pred_revised.append(0)
#%%             
cm = confusion_matrix(y_test, y_pred_revised)
print(cm)
# %%
report = classification_report(y_test, y_pred_revised)
print(report) # 86% accuracy
# %%


# ----------------------------------------------------------------------
#%%
'''Stemming allows the computer to group together words according to their stem. For instance: “walk,” “walked” and “walking.”

Lemmatization is a bit more complex in that the computer can group together words that do not have the same stem, but still have the same inflected meaning. Grouping the word “good” with words like “better” and “best” is an example of lemmatization.'''
## spaCy 
import spacy
# en_core_web_sm is a pre built pipeline. We can build our own as well by adding the components we need. 
nlp = spacy.load("en_core_web_sm")
doc = nlp('I love this package. This is turning out to be pretty useful') 
print([(w.text, w.pos_) for w in doc]) # pos - 'part of speech' tagging. 
# %%
token0 = doc[0]
# %%
# %%
# will split the input into various sentences seperated by '.' or period. 
for sentence in doc.sents:
    print(sentence)
# %%
# spaCy components. These components are part of the pipeline. 
# ner - name entity recognization. finds/tags names and entities in docs.
doc = '''Inflation rose again in April, continuing a climb that has pushed consumers to the brink and is threatening the economic expansion, the Bureau of Labor Statistics reported Wednesday.

The consumer price index, a broad-based measure of prices for goods and services, increased 8.3% from a year ago, higher than the Dow Jones estimate for an 8.1% gain. That represented a slight ease from March’s peak but was still close to the highest level since the summer of 1982.

Removing volatile food and energy prices, so-called core CPI still rose 6.2%, against expectations for a 6% gain, clouding hopes that inflation had peaked in March.

The month-over-month gains also were higher than expectations — 0.3% on headline CPI versus the 0.2% estimate and a 0.6% increase for core, against the outlook for a 0.4% gain.

The price gains also meant that workers continued to lose ground. Real wages adjusted for inflation decreased 0.1% on the month despite a nominal increase of 0.3% in average hourly earnings. Over the past year, real earnings have dropped 2.6% even though average hourly earnings are up 5.5%.

Inflation has been the single biggest threat to a recovery that began early in the Covid pandemic and saw the economy in 2021 stage its biggest single-year growth level since 1984. Rising prices at the pump and in grocery stores have been one problem, but inflation has spread beyond those two areas into housing, auto sales and a host of other areas.

Federal Reserve officials have responded to the problem with two interest rate hikes so far this year and pledges of more until inflation comes down to the central bank’s 2% goal. However, Wednesday’s data shows that the Fed has a big job ahead.'''
# %%
import spacy
nlp = spacy.load("en_core_web_sm")
processed_doc = nlp(doc)
# %%
nouns = []
numeric_token = []
for token in processed_doc:
    if token.pos_ == "NOUN": 
        nouns.append(token)
    if token.pos_ == "NUM":
        numeric_token.append(token)
#%% 

# %%
