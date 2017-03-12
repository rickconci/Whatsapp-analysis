
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

f = open("clara_chat.txt")


# In[3]:

lines = f.readlines()


# In[4]:

lines2 = []


# In[5]:

for line in lines:
    try:
        tokens = line.split(":")
        t1 = ":".join(tokens[0:2])
        t2 = tokens[3].strip()
        text = tokens[4:]
        text = ":".join(text).strip()
        lines2.append([t1, t2, text])
    except:
        print("couldn't process")
        continue
# lines[0]


# In[6]:

df = pd.DataFrame(lines2)


# In[7]:

df.columns = ['created', 'sender', 'text']
import numpy as np
df = df[np.invert(df['text'].str.contains('image omitted'))]


# In[8]:

df['len'] = df['text'].str.len()


# In[9]:

for sender in df['sender'].unique():
    print(sender,df[df['sender']==sender]['len'].mean())


# In[10]:

df['created'] = pd.to_datetime(df['created'])


# In[11]:

df = df.set_index('created')


# In[12]:

get_ipython().magic('matplotlib inline')
tss = {}
for sender in df['sender'].unique():
    tmp = df[df['sender']==sender]
    tss[sender.strip()] = tmp.resample('D', how='count')['sender']
    tmp.resample('D', how='count')['sender'].plot(figsize=(15,5))


# In[13]:

rs = pd.DataFrame(tss)


# In[14]:

rs['diff'] = rs['Clara'] - rs['Riccardo Conci']


# In[15]:

tmp = rs[rs.index>'2015-10-01']
tmp = tmp[tmp.index<'2016-10-01']
tmp['Clara'].plot(figsize=(15,5))
tmp['Riccardo Conci'].plot(figsize=(15,5))
tmp['diff'].plot(figsize=(15,5))


# In[16]:

from wordcloud import WordCloud
import seaborn as sns


# In[17]:

def plotText(text):
    wordcloud = WordCloud().generate(text)
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis("off")


# In[18]:

import matplotlib.pyplot as plt
for sender in ['Riccardo Conci', 'Clara']:
    text = " ".join(df[df['sender']==sender]['text'].tolist())
    print(sender)
    plotText(text)
    plt.show()


# In[19]:

def getPolarity(text):
    import textblob
    o = textblob.TextBlob(text)
    return o.sentiment[0]


# In[20]:

df['polarity'] = df['text'].apply(getPolarity)


# In[21]:

for sender in df['sender'].unique():
    print(sender)
    print(df[df['sender']==sender]['polarity'].mean())


# In[22]:

df[df['polarity']<0]['sender'].value_counts()


# In[23]:

for sender in ['Riccardo Conci', 'Clara']:
#     plt.show()
    df[df['sender']==sender]['polarity'].hist(bins=20)
    print(sender, df[df['sender']==sender]['polarity'].describe())


# In[24]:

df = df[df['sender'].isin(['Riccardo Conci', 'Clara'])]


# In[25]:

sns.violinplot(data=df, x='sender', y='polarity')


# In[26]:

import textblob
o = textblob.TextBlob(text.lower())
print (o.word_counts["honey"])


# In[27]:

print(len(df[df['text'].str.contains('honey', case=False)]))


# In[28]:

df['sender'].unique()


# In[29]:

import numpy as np
print(len(df))
df = df[np.invert(df['text'].str.contains('image omitted'))]
print(len(df))
df1 = df[df['sender']=='Clara']
df2 = df[df['sender']=='Riccardo Conci']


# In[30]:

text1 = " ".join(df1['text'].tolist())
text2 = " ".join(df2['text'].tolist())


# In[31]:

from nltk.tokenize import RegexpTokenizer
from nltk import ngrams
tokenizer = RegexpTokenizer(r'\w+')
from collections import Counter


# In[32]:

text1 = tokenizer.tokenize(text1)
text2 = tokenizer.tokenize(text2)


# In[33]:

def compare(text1, text2, N):
    grams1 = Counter(list(ngrams(text1, N)))
    grams2 = Counter(list(ngrams(text2, N)))
    d1 = pd.DataFrame.from_dict(grams1, orient='index').reset_index()
    d2 = pd.DataFrame.from_dict(grams2, orient='index').reset_index()    
    d1.columns = ['term', 'count']
    d2.columns = ['term', 'count']    
    d1['frequency'] = d1['count'].astype(float)/len(d1)
    d2['frequency'] = d2['count'].astype(float)/len(d2)    
    d = pd.merge(d1, d2, on='term')
    d['diff_freq'] = d['frequency_x'] - d['frequency_y']
    print(grams1.most_common(10))
    print("\n")
    print(grams2.most_common(10))
    return d


# In[34]:

d = compare(text1, text2, 3)


# In[35]:

d.sort('diff_freq', inplace=True)


# In[36]:

d[d['term'].astype(str).str.contains('😘', case=False)]
#words to search: love, know, sure, care, sorry, why, 


# In[37]:

d.sort('count_y')


# In[38]:

len(df1[df1['text'].str.contains("😉")])


# In[39]:

len(df2[df2['text'].str.contains("😉")])


# In[40]:

df['created'] = pd.to_datetime(df['created'])


# In[ ]:

df = df.set_index('created')


# In[ ]:

df.resample('W', how='mean')['polarity'].plot()


# In[41]:

df.resample('W', how='mean')['polarity']


# In[ ]:




# In[ ]:



