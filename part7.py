from nltk.book import *

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import brown


words = brown.words()  # words in whole corpus
N = len(words)  # number of words in corpus


pituudet = [
    len(w) for w in words if w.isalpha()
]  # lengths of  words that contain only letters
fdist = FreqDist(pituudet)
N2 = len(fdist)  # number of different wordlengths


y = fdist.most_common()  # ordering
y2 = [y[i][1] for i in range(N2)]  # list containing number of occasions
x = [(j + 1) for j in range(N2)]  # list of ranks


y3 = np.array(y2)  # list to array
y4 = y3 * (1 / N)  # numbers to proportions
y5 = np.log(y4)  # log of frequence

x2 = np.log(x)  # log of rank

# %% Python visualization with pyplot
scale = 2
plt.figure(figsize=(6 * scale, 12 * scale), dpi=80)
# plt.figure(figsize=(10, 10))  # to increase the plot resolution
# plt.ylabel("Log Frequency", fontsize=24)
# plt.xlabel("Log Ranks", fontsize=24)
plt.ylabel("Log Frequency", fontsize=12 * scale)
plt.xlabel("Log Ranks", fontsize=12 * scale)
plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)
plt.suptitle("7 Length of tokens", fontsize=24)
# plt.xticks(rotation=90)  # to rotate x-axis values


plt.plot(x2, y5)
# plt.savefig('save as png.png')
plt.show()
