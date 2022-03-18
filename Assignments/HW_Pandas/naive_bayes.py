
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
# %%
digits = load_digits()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3 )
# %%
model1 = LogisticRegression()
model1.fit(X_train, y_train)
model1.score(X_test, y_test)
# %%
model2 = SVC()
model2.fit(X_train, y_train)
# %%
model2.score(X_test, y_test)
# %%
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)

# %%
model3.score(X_test, y_test)
# %%
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)


# %%
#for train_index, test_index in kf.split(digits.data):
#    kk
# %%
    def get_score(model, X_train, X_test, y_train, y_test):
        m = model
        m.fit(X_train, y_train)
        return m.score(X_test, y_test)
    


# %%
get_score(SVC(), X_train, X_test, y_train, y_test)

# %%

# %%
import pandas as pd
df = pd.read_csv('spam.csv', sep= ',')

# %%
df['spam'] = df['Category'].apply(lambda x : 1 if x == 'spam' else 0)

# %%
df.head(10)
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.25)

# %%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
# %%
X_train_count = cv.fit_transform(X_train.values)
X_train_count.toarray()
# %%
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)
# %%
email = ['Hi, how are you this morning?',
         'Please buy this!!!!!!Free pizza']
email_test = cv.fit_transform(email)
# email_test.toarray() - Not sure if this is needed or not. 
# %%
model.predict(email_test)
# %%
X_test_count  = cv.transform(X_test)
# %%
model.score(X_test_count, y_test)

# %%
