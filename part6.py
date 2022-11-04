# NLP Projekti 1: Zipz law, kohta 6
# Susanna

from nltk.corpus import brown
from sklearn.utils import shuffle
from nltk import bigrams
import nltk
import numpy as np

# import nltk
# nltk.download('brown')
"""Repeat 4) when using the big-gram, highlighting the frequency of tag pairs and their rankings."""

# brown.categories()

# -----Split the brown corpus into 8 samples-----

# First suffle the file IDs
newbrown = brown
sb = shuffle(newbrown.fileids())

# The the subcorpus's IDs: C1,C2,C3,C4,C5,C6,C7,C8
id1 = sb[0:62]
id2 = sb[62:124]
id3 = sb[124:186]
id4 = sb[186:248]
id5 = sb[248:310]
id6 = sb[310:372]
id7 = sb[372:434]
id8 = sb[434:]

c12 = id1 + id2  # {C1,C2}
c1234 = id1 + id2 + id3 + id4  # {C1,C2,C3,C4}
c_all = id1 + id2 + id3 + id4 + id5 + id6 + id7 + id8  # {C1,C2,C3,C4,C5,C6,C7,C8}

# I could have done this just by making a function xD but well

########### ------------ C1 -------------- ##########
# first initiolize: first document's words
C1_words = brown.words(fileids=id1[0])
# C1_words = list(bigrams(brown.words(fileids=id1[0])))

# add the other documents's words
for i in id1[1:]:
    C1_words = C1_words + brown.words(fileids=i)
    # C1_words = C1_words +list(bigrams(brown.words(fileids=i)))

# only alphabets
sanat2 = [w for w in C1_words if w.isalpha()]

# take bigram
C1_words = list(bigrams(sanat2))
len(C1_words)  # 121789

# Get the freqvences of words
# freqC1 = nltk.FreqDist(w.lower() for w in C1_words)
freqC1 = nltk.FreqDist(w for w in C1_words)
# len 78539

# Take freqvencies from cormps C1 to a list
keysC1 = list(freqC1.keys())
fC1 = []

for i in keysC1:
    fC1.append(freqC1[i])

fC1.sort(reverse=True)

##################----- {C1, C2} ------###############
# C12_words = list(bigrams(brown.words(fileids=c12[0])))
C12_words = brown.words(fileids=c12[0])
# add the others
for i in c12[1:]:
    C12_words = C12_words + brown.words(fileids=i)

sanat2 = [w for w in C12_words if w.isalpha()]

# take bigram
C12_words = list(bigrams(sanat2))
# len 243660
len(C12_words)

# Get the freqvences of words
freqC12 = nltk.FreqDist(w for w in C12_words)
# len 142399
# Take freqvencies from cormpus {C1, C2} to a list

keysC12 = list(freqC12.keys())  # get the keys
fC12 = []

for i in keysC12:
    fC12.append(freqC12[i])
fC12.sort(reverse=True)

##################{C1, C2, C3, C4}################3

# ---- Select the subcorpus:
C1234_words = brown.words(fileids=c1234[0])
# add the others
for i in c1234[1:]:
    C1234_words = C1234_words + brown.words(fileids=i)

sanat2 = [w for w in C1234_words if w.isalpha()]

# take bigram
C1234_words = list(bigrams(sanat2))

len(C1234_words)  # 487000
# Get the freqvences of words
freqC1234 = nltk.FreqDist(w for w in C1234_words)
# len 254347

# Take freqvencies from freqDis {C1, C2, C3, C4}
keysC1234 = list(freqC1234.keys())
fC1234 = []

for i in keysC1234:
    fC1234.append(freqC1234[i])

fC1234.sort(reverse=True)

############# {C1, C2, C3, C4, C5, C6, C7, C8}###########

# Get the freqvences of words
sanat2 = [w for w in brown.words() if w.isalpha()]
# len 981716
bigram_all = list(bigrams(sanat2))
# len 981715
freqC_all = nltk.FreqDist(w for w in bigram_all)
# len 452399

# Take freqvencies from cormpus {all}

keysCall = list(freqC_all.keys())  # get keys
fCall = []  # make list for frecvencies

for i in keysCall:
    fCall.append(freqC_all[i])
fCall.sort(reverse=True)

######## .....  plot in same picture .... ########

# X-axel, 'rank'
rank100 = list(range(1, 101))

import matplotlib.pyplot as plt

plt.plot(
    rank100,
    fC1[:100],
    "b-",
    rank100,
    fC12[:100],
    "r-",
    rank100,
    fC1234[:100],
    "g-",
    rank100,
    fCall[:100],
    "c-",
)
plt.xlabel("Rank")
plt.ylabel("Frequency (bigram)")
plt.title("Zipf's law of different size corpuses (top 100)")
plt.legend(["C1", "{C1,C2}", "{C1, C2, C3, C4}", "All"], loc="upper right")
plt.show()

###### LOG LOG plots ###

logx = np.log(rank100)

y1 = np.array(fC1)
y2 = np.array(fC12)
y3 = np.array(fC1234)
y4 = np.array(fCall)

# Frequency log
log_y1 = np.log(y1)
log_y2 = np.log(y2)
log_y3 = np.log(y3)
log_y4 = np.log(y4)

plt.plot(
    logx,
    log_y1[:100],
    "b-",
    logx,
    log_y2[:100],
    "r-",
    logx,
    log_y3[:100],
    "g-",
    logx,
    log_y4[:100],
    "c-",
)
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log,bigram)")
plt.title("log-log Zipf's law of different size corpuses (top 100)")
plt.legend(["C1", "{C1,C2}", "{C1, C2, C3, C4}", "All"], loc="upper right")
plt.show()
