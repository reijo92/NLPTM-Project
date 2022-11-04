# NLP Projekti 1: Zipz law, kohta 4
# Susanna

import matplotlib.pyplot as plt
from nltk.corpus import brown
from sklearn.utils import shuffle
import nltk
import numpy as np

# import nltk
# nltk.download('brown')
"""We want to test how the behavior of Zipf’s law when combining corpuses.
For this purpose, split randomly Brown corpus into 8 roughly equal samples (C1, C2, ..C8).
 Draw the zipfs’ law for corpus 1, then for corpus formed by concatenating {C1, C2},
 then {C1,C2, C3, C4}, then {C1, C2, C3, C4, C5, C6, C7, C8}
 (each time we concatenate by (sub) corpus of same size.
  Draw all the illustrations on the same plot."""

# brown.categories()

# Split the brown corpus into 8 samples
# Täää
newbrown = brown
sb = shuffle(newbrown.fileids())

id1 = sb[0:62]
id2 = sb[62:124]
id3 = sb[124:186]
id4 = sb[186:248]
id5 = sb[248:310]
id6 = sb[310:372]
id7 = sb[372:434]
id8 = sb[434:]

c12 = id1 + id2
c1234 = id1 + id2 + id3 + id4
c_all = id1 + id2 + id3 + id4 + id5 + id6 + id7 + id8

# Draw zipfs law

########### --- C1 --- ##########
# first initiolize
C1_words = brown.words(fileids=id1[0])
# add the others
for i in id1[1:]:
    C1_words = C1_words + brown.words(fileids=i)

sanat2 = [w for w in C1_words if w.isalpha()]
len(C1_words)  # 144088
len(sanat2)  # 121975
# Get the freqvences of words

freqC1 = nltk.FreqDist(w.lower() for w in sanat2)
# len 13557

# Take  freqvencies from cormus C1
keysC1 = list(freqC1.keys())
fC1 = []

for i in keysC1:
    fC1.append(freqC1[i])

fC1.sort(reverse=True)

################## {C1, C2} ###############
C12_words = brown.words(fileids=c12[0])
# add the others
for i in c12[1:]:
    C12_words = C12_words + brown.words(fileids=i)

sanat2 = [w for w in C12_words if w.isalpha()]
len(C12_words)  # 288516~
len(sanat2)  # 243791
# Get the freqvences of words

freqC12 = nltk.FreqDist(w.lower() for w in sanat2)
# len 20549
# Take freqvencies from cormpus {C1, C2}

keysC12 = list(freqC12.keys())
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
len(sanat2)  # 486946

# Get the freqvences of words
freqC1234 = nltk.FreqDist(w.lower() for w in sanat2)
# len 28906

# Take freqvencies from freqDis {C1, C2, C3, C4}
keysC1234 = list(freqC1234.keys())
fC1234 = []

for i in keysC1234:
    fC1234.append(freqC1234[i])

fC1234.sort(reverse=True)

############# {C1, C2, C3, C4, C5, C6, C7, C8}###########

# Get the freqvences of words: 1161192
sanat2 = [w for w in brown.words() if w.isalpha()]
# len 981716
freqC_all = nltk.FreqDist(w.lower() for w in sanat2)
# len 40234
# Take freqvencies from cormpus {all}

keysCall = list(freqC_all.keys())  # get keys
fCall = []  # make list for frecvencies

for i in keysCall:
    fCall.append(freqC_all[i])
fCall.sort(reverse=True)

######## .....  plot in same picture .... ########

# X-axel, 'rank'
rank100 = list(range(1, 101))


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
plt.ylabel("Frequency (words)")
plt.title("Zipf's law of different size corpuses (top 100)")
plt.legend(["C1", "{C1,C2}", "{C1, C2, C3, C4}", "All"], loc="upper right")
plt.show()


# LOG LOG PLOT ###


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
plt.ylabel("Frequency (log, words)")
plt.title("log-log Zipf's law of different size corpuses (top 100)")
plt.legend(["C1", "{C1,C2}", "{C1, C2, C3, C4}", "All"], loc="upper right")
plt.show()
