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
# running through gensim preprocessing.  - This converts each sentence into an array of words and removes unnecessary words from the sentence.
review_text = df['review'].apply(lambda x : gensim.utils.simple_preprocess(x))
# %%
model = gensim.models.Word2Vec(
        window = 5,
        min_count = 3,
        workers = 4
)
#%%
# building a vocabulary now. This contains a list of all the different words. 
vocab = model.build_vocab(review_text, progress_per=1000)
# %%
# corpus_count is the total examples in review_text. 
model.train(review_text, total_examples=model.corpus_count, epochs = model.epochs)
# %%
model.wv.most_similar('good')
# %%
model.wv.get_index('good')

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
padded_revs
len(padded_revs[0]) # 2500
len(padded_revs[49999]) # 2500
# %%
# Now we will describe the number of features to develop a relation between the words. 
features = 10
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
print(report)
# %%
