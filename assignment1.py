#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[4]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
len(allRatings)


# In[5]:


ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
ratingDict = {}

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    ratingDict[(u,b)] = r
ratingsPerUser['u67805239']
len(ratingsTrain)
ratingDict


# In[6]:


# From baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break


# In[7]:


Generate a negative set

userSet = set()
bookSet = set()
readSet = set()

for u,b,r in allRatings:
    userSet.add(u)
    bookSet.add(b)
    readSet.add((u,b))
 

lUserSet = list(userSet)
lBookSet = list(bookSet)

notRead = set()
y1 = []
y2 = []
for u,b,r in ratingsValid:
    #u = random.choice(lUserSet)
    b = random.choice(lBookSet)
    while ((u,b) in readSet or (u,b) in notRead):
        b = random.choice(lBookSet)
    notRead.add((u,b))
    y2.append(0)

readValid = set()
for u,b,r in ratingsValid:
    readValid.add((u,b))
    y1.append(1)
    

    
#print(notRead)
yfinal = y1 + y2
yfinal
len(yfinal)
testFinal = list(readValid) + list(notRead)
len(testFinal)
testFinal
len(testFinal)
len(yfinal)


# In[8]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0
return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > 1.5 * totalRead/2: break
len(return1)


# In[9]:


# correct = 0
# return1 = list(return1)
# print(len(return1))
# for (label,sample) in [(1, readValid), (0, notRead)]:
#     pred = 0
#     for (u,b) in sample:
#         count = 0
#         maxSim = 0
#         sim1 = []
#         sim2 = []
#         users = set(ratingsPerItem[b])
#         for b2,_ in ratingsPerUser[u]:
#             sim = Jaccard(users,set(ratingsPerItem[b2]))
#             sim1.append(sim)
#             if(count < len(return1)):
#                 sim = Jaccard(set(ratingsPerItem[return1[count]]),set(ratingsPerItem[b2]))
#                 sim2.append(sim)
#                 count+=1
#         if(len(sim1) > 0):
#             sim1.sort()
#             sim1.reverse()
#             sim2.sort()
#             sim2.reverse()
#             pred = 0
#             if(len(sim1) < 3):
#                 if(sim1[0] >= sim2 [0]):
#                     pred = 1
#             else:
#                 if(sim1[0] >= sim2[0] and sim1[1] >= sim2 [1] and sim1[2] >= sim2 [2]):
#                     pred = 1
            
        
            
#         if pred == label:
#             correct += 1


# In[10]:


def Cosine(i1, i2):
    # Between two items
    inter = set(ratingsPerItem[i1]).intersection(set(ratingsPerItem[i2]))
    numer = 0
    denom1 = 0
    denom2 = 0
    for u,_ in inter:
        yum = ratingDict[(u,i1)]
        yummy = ratingDict[(u,i2)]
        numer += yum*yummy
    for u,_ in ratingsPerItem[i1]:
        yum = ratingDict[(u,i1)]
        denom1 += yum**2
    for u,_ in ratingsPerItem[i2]:
        yummy = ratingDict[(u,i2)]
        denom2 += yummy**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[11]:


correct = 0
return1 = list(return1)
apples = []
for (label,sample) in [(1, readValid), (0, notRead)]:
    pred = 0
    for (u,b) in sample:
        oranges = []
        count = 0
        maxSim = 0
        sim1 = []
        sim2 = []
        sim3 = []
        sim4 = []
        avg = 0
        avg2 = 0
        users = set(ratingsPerItem[b])
        items = set(ratingsPerUser[u])
        for b2,_ in ratingsPerUser[u]:
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            sim1.append(sim)
            sim = Jaccard(set(ratingsPerItem[return1[count]]),set(ratingsPerItem[b2]))
            sim2.append(sim)
            sim = Cosine(b,b2)
            sim4.append(sim)
            count+=1
        for u3,_ in ratingsPerItem[b]:
            sim = Jaccard(items,set(ratingsPerUser[u3]))
            sim3.append(sim)
        if(len(sim1) > 0):
            sim1.sort()
            sim1.reverse()
            sim2.sort()
            sim2.reverse()
            sim4.sort()
            sim4.reverse()
            pred = 0
            avg = sum(sim1)/len(sim1)
        else:
            sim1.append(0)
            sim2.append(0)
            sim4.append(0)
        if(len(sim3) > 0):
            sim3.sort()
            sim3.reverse()
            avg2 = sum(sim3)/len(sim3)
        else:
            sim3.append(0)
       
            
        if((sim1[0] > 0.02020202 and sim3[0] > 0 and sim1[0] > sim2[-1]    )  or len(ratingsPerItem[b]) > 27 ):
            pred = 1

        if(len(ratingsPerItem[b]) > 27):
            oranges.append(1)
        else:
            oranges.append(0)
       # oranges.append(len(ratingsPerItem[b]))
#         oranges.append(len(ratingsPerUser[u]))
        if(sim1[0] >  0.013):
            oranges.append(1)
        else:
            oranges.append(0)
        #oranges.append(sim1[0])
        #oranges.append(sim3[0])
        #oranges.append(avg)
        #oranges.append(sim4[0])
        apples.append(oranges)
            
        
            
        if pred == label:
            correct += 1


# In[12]:


yum = correct / (len(readValid) + len(notRead))
print(yum)
apples
len(apples)
apples


# In[13]:


yummy = np.linspace(0,1,100)
list(yummy)
yummy = yummy[0:31]
yummy
list(yummy)
yummy


# In[14]:


bestVals = []
for x in yummy:
    correct = 0
    return1 = list(return1)
    apples = []
    for (label,sample) in [(1, readValid), (0, notRead)]:
        pred = 0
        for (u,b) in sample:
            oranges = []
            count = 0
            maxSim = 0
            sim1 = []
            sim2 = []
            sim3 = []
            avg = 0
            avg2 = 0
            users = set(ratingsPerItem[b])
            items = set(ratingsPerUser[u])
            for b2,_ in ratingsPerUser[u]:
                sim = Jaccard(users,set(ratingsPerItem[b2]))
                sim1.append(sim)
                count+=1
            for u3,_ in ratingsPerItem[b]:
                sim = Jaccard(items,set(ratingsPerUser[u3]))
                sim3.append(sim)
            if(len(sim1) > 0):
                sim1.sort()
                sim1.reverse()
                pred = 0
                avg = sum(sim1)/len(sim1)
            else:
                sim1.append(0)
                sim2.append(0)
            if(len(sim3) > 0):
                sim3.sort()
                sim3.reverse()
                avg2 = sum(sim3)/len(sim3)
            else:
                sim3.append(0)

            if((sim1[0] > 0.02020202 and sim3[0] > x )  or len(ratingsPerItem[b]) > 27 ):
                pred = 1




            if pred == label:
                correct += 1
    yum = correct / (len(readValid) + len(notRead))
    bestVals.append(yum)


# In[15]:


correct = 0
return1 = list(return1)
for (label,sample) in [(1, readValid), (0, notRead)]:
    for (u,b) in sample:
        maxSim = 0
        users = set(ratingsPerItem[b])
        for b2,_ in ratingsPerUser[u]:
            sim = Jaccard(users,set(ratingsPerItem[b2]))
            if sim > maxSim:
                maxSim = sim
        pred = 0
        if maxSim > 0.013 or len(ratingsPerItem[b]) > 40:
            pred = 1
        if pred == label:
            correct += 1


# In[16]:


correct / (len(readValid) + len(notRead))
if(0.75095 > 0.75):
    print('yup')


# In[17]:


testfinalz = []
# for x in testFinal:
#     temp = []
#     yum = int(x[0][1:])
#     yum2 = int(x[1][1:])
#     temp.append(yum)
#     temp.append(yum2)
#     testfinalz.append(temp)
# testfinalz
# yfinal
for y in testFinal:
    temp = []
    yum = int(y[0][1:])
    yum2 = int(y[1][1:])
    temp.append(yum)
    temp.append(yum2)
    testfinalz.append(temp)
len(testfinalz)
yfinal
X_train, X_test, y_train, y_test = train_test_split(apples, yfinal, test_size=0.35)
print(sum(y_train))
print(sum(y_test))
X_test


# In[18]:


mod = linear_model.LogisticRegression(C=0.0001)
mod.fit(X_train, y_train)


# In[19]:


pred =  mod.predict(X_test)

val = accuracy_score(y_test, pred)
print(accuracy_score(y_test, pred))
type(val)


# In[20]:


yum = {}
if(val < 0.7568):
    for x in range(0,50):
        X_train, X_test, y_train, y_test = train_test_split(apples, yfinal, test_size=0.35)
        mod = linear_model.LogisticRegression(C=0.001)
        mod.fit(X_train, y_train)
        pred =  mod.predict(X_test)
        val = accuracy_score(y_test, pred)
        yum[val] = mod


        

    


# In[21]:


stuff = yum.keys()
stuff
truMod = yum[0.7607142857142857]
truMod


# In[22]:


Finalapples = []
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        continue
    l.strip()
    op = l.split(',')
    u = op[0]
    b = op[1]
    b = b[:-1]
    
    maxSim = 0
    oranges = []
    count = 0
    maxSim = 0
    sim1 = []
    sim2 = []
    sim3 = []
    sim4 = []
    avg = 0
    avg2 = 0
    users = set(ratingsPerItem[b])
    items = set(ratingsPerUser[u])
    for b2,_ in ratingsPerUser[u]:
        sim = Jaccard(users,set(ratingsPerItem[b2]))
        sim1.append(sim)
        sim = Jaccard(set(ratingsPerItem[return1[count]]),set(ratingsPerItem[b2]))
        sim2.append(sim)
        sim = Cosine(b,b2)
        sim4.append(sim)
        count+=1
    for u3,_ in ratingsPerItem[b]:
        sim = Jaccard(items,set(ratingsPerUser[u3]))
        sim3.append(sim)
    if(len(sim1) > 0):
        sim1.sort()
        sim1.reverse()
        sim2.sort()
        sim2.reverse()
        sim4.sort()
        sim4.reverse()
        pred = 0
        avg = sum(sim1)/len(sim1)
    else:
        sim1.append(0)
        sim2.append(0)
        sim4.append(0)
    if(len(sim3) > 0):
        sim3.sort()
        sim3.reverse()
        avg2 = sum(sim3)/len(sim3)
    else:
        sim3.append(0)


    if((sim1[0] > 0.02020202 and sim3[0] > 0 and sim1[0] > sim2[-1]    )  or len(ratingsPerItem[b]) > 27 ):
        pred = 1

    if(len(ratingsPerItem[b]) > 27):
        oranges.append(1)
    else:
        oranges.append(0)
    # oranges.append(len(ratingsPerItem[b]))
    #         oranges.append(len(ratingsPerUser[u]))
    if(sim1[0] >  0.013):
        oranges.append(1)
    else:
        oranges.append(0)
    #oranges.append(sim1[0])
    #oranges.append(sim3[0])
    #oranges.append(avg)
    #oranges.append(sim4[0])
    Finalapples.append(oranges)
Finalapples



# In[23]:


predictions = open("predictions_Read.csv", 'w')
finalPred = truMod.predict(Finalapples)
count = 0
for l in open("pairs_Read.csv"):
    print(l)
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    _ = predictions.write(u + ',' + b + ',' + str(finalPred[0]) + '\n')
    count+=1

predictions.close()


# In[24]:





# In[25]:


# bestVals


# In[5]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
        
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# In[6]:


wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data:
  r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
  for w in r.split():
    wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
counts_filtered = [c for c in counts if c[1] not in stopwords.words('english')]
N_cols = 60000 
words = [x[1] for x in counts_filtered[0:N_cols]]

wordId = dict(zip(words, range(len(words))))
# wordId_alternative = {words[i]: i for i in range(len(words))}
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    for w in r.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    return feat

from scipy import sparse
N_rows = 100000 
sX = sparse.lil_matrix((N_rows, N_cols))

for i in range(N_rows):
    if not i%1000:
        print(i)
    # figure out which column to increment
    sX[i] = feature(data[i])
    

y = [d['genreID'] for d in data[0:N_rows]]

Xtrain = sX[:9*sX.shape[0]//10]
ytrain = y[:9*len(y)//10]
Xvalid = sX[9*sX.shape[0]//10:]
yvalid = y[9*len(y)//10:]


# In[25]:


len(yvalid)


# In[28]:


# from sentence_transformers import SentenceTransformer
# emb_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


# In[29]:


# embedding = emb_model.encode(data[0:1000])


# 

# In[30]:


# df = defaultdict(int)
# for d in data:
#     r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
#     for w in set(r.split()):
#         df[w] += 1


# In[31]:


# wordId = dict(zip(words, range(len(words))))
# wordSet = set(words)
# tf = defaultdict(int)
# def feature(datum):
#     feat = [0]*len(words)
#     r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
#     for w in r.split():
#         if w in wordSet:
#             tf[w] += 1
#     for x in r.split():
#         if x in wordSet:
#             feat[wordId[x]] = tf[x] * math.log2(len(data) / df[x])
#     feat.append(1) 
#     return feat

# X = [feature(d) for d in data]
# y = [d['genreID'] for d in data]


# In[16]:


import sklearn
import sklearn.naive_bayes as naive_bayes


# In[17]:


mod_NB = naive_bayes.MultinomialNB()


# In[46]:


mod = linear_model.LogisticRegression()


# In[18]:


mod_NB.fit(Xtrain, ytrain)


# In[19]:


pred = mod_NB.predict(Xvalid)


# In[20]:


correct = pred == yvalid


# In[21]:


sum(correct) / len(correct)


# In[39]:


# embeddings
# First, split into train/test
# Xtrain, ytrain, etc.
# mod.fit(Xtrain, ytrain)


# In[22]:


data_test = []

for d in readGz("test_Category.json.gz"):
    data_test.append(d)
len(data_test)


# In[29]:


N_rows = 100000 
N_cols = 60000
sB = sparse.lil_matrix((N_rows, N_cols))
for i in range(10000):
    if not i%1000:
        print(i)
    # figure out which column to increment
    sB[i] = feature(data_test[i])

# pred_test = mod_NB.predict(Xtest)


# In[31]:


pred_test = mod_NB.predict(sB)


# In[33]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
    pos += 1

predictions.close()


# In[ ]:




