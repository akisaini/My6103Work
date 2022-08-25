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
