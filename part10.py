from nltk.book import *

import nltk
from nltk.corpus import brown

from matplotlib import pyplot as plt
import numpy as np

from nltk.corpus import wordnet as wn


words = brown.words()  # words of corpus
words2 = [w for w in words if w.isalpha()]  # only words that contain only letters
N = len(words2)  # number of words


if N < 5:
    print("too small file")
    quit()


result = []  # list of locations of homonyms
# 1 if a homonym, 0 otherwise
result2 = []  # list on number of neighbouring homonyms


# homonym codes 1 or 0 for words 1-4
if len(wn.synsets(words2[0])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[1])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[2])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[3])) > 1:
    result.append(1)
else:
    result.append(0)


# number on homonyms of words 1 and 2
if len(wn.synsets(words2[0])) > 1:
    result2.append(result[1] + result[2])
if len(wn.synsets(words2[1])) > 1:
    result2.append(result[0] + result[2] + result[3])


# the same for rest of the words
for i in range(4, N):
    n = len(wn.synsets(words2[i]))
    if n > 1:
        result.append(1)
    else:
        result.append(0)
    if result[i - 2] > 0:
        result2.append(result[i - 4] + result[i - 3] + result[i - 1] + result[i])


# the amounts of neihgbouring homonyms for last 2 words
if len(wn.synsets(words2[N - 2])) > 1:
    result2.append(result[N - 4] + result[N - 3] + result[N - 1])
if len(wn.synsets(words2[N - 1])) > 1:
    result2.append(result[N - 3] + result[N - 2])


# creating a histogram
result3 = np.array(result2)  # list to array

scale = 2
fig, axis = plt.subplots(figsize=(6 * scale, 12 * scale), dpi=80)
axis.hist(
    result3, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], density="True", edgecolor="black"
)
plt.ylabel("Percentage", fontsize=12 * scale)
plt.xlabel("Homonym amount in neighbourhood", fontsize=12 * scale)

plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)
plt.suptitle("10. Neighbouring homonyms", fontsize=12 * scale)


plt.show()




words = brown.words()  # words of corpus
words2 = [w for w in words if w.isalpha()]  # only words that contain only letters
N = len(words2)  # number of words


if N < 5:
    print("too small file")
    quit()


result = []  # list of locations of homonyms
# 1 if a homonym, 0 otherwise
result2 = []  # list on number of neighbouring homonyms


# homonym codes 1 or 0 for words 1-4
if len(wn.synsets(words2[0])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[1])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[2])) > 1:
    result.append(1)
else:
    result.append(0)

if len(wn.synsets(words2[3])) > 1:
    result.append(1)
else:
    result.append(0)


# number on homonyms of words 1 and 2
if len(wn.synsets(words2[0])) > 1:
    result2.append(result[1] + result[2])
if len(wn.synsets(words2[1])) > 1:
    result2.append(result[0] + result[2] + result[3])

# the same for rest of the words
for i in range(4, N):
    n = len(wn.synsets(words2[i]))
    if n > 1:
        result.append(1)
    else:
        result.append(0)
    if result[i - 2] > 0:
        result2.append(result[i - 4] + result[i - 3] + result[i - 1] + result[i])


# the amounts of neihgbouring homonyms for last 2 words
if len(wn.synsets(words2[N - 2])) > 1:
    result2.append(result[N - 4] + result[N - 3] + result[N - 1])
if len(wn.synsets(words2[N - 1])) > 1:
    result2.append(result[N - 3] + result[N - 2])
# creating a histogram
result3 = np.array(result2)  # list to array
# fig, axis = plt.subplots(figsize=(10, 5))
fig, axis = plt.subplots(figsize=(6 * scale, 12 * scale), dpi=80)

plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)

axis.hist(result3, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], edgecolor="black")
plt.show()


words = brown.words()  # words of corpus
words2 = [w for w in words if w.isalpha()]  # only words that contain only letters


# creating a vector indicating homonymity
result = []
for i in range(len(words2)):
    n = len(wn.synsets(words2[i]))
    if n > 1:
        result.append(1)
    else:
        result.append(0)


# choosing only homonyms
words3 = [words2[i] for i in range(len(words2)) if result[i] > 0]


# creating part-of-speech-tags
tagged = nltk.pos_tag(words3)


# isolating the tag part
tags = [tagged[i][1] for i in range(len(words3))]


# frequency distribution of tags
fdist = FreqDist(tags)
fdist2 = fdist.most_common()  # ordering


# creating a histogram
X = [fdist2[i][0] for i in range(len(fdist2))]  # different tags
Y = [fdist2[i][1] for i in range(len(fdist2))]  # their amounts
fig, axis = plt.subplots(figsize=(6 * scale, 12 * scale), dpi=80)
plt.xticks(rotation=90)  # to rotate x-axis values
plt.xticks(fontsize=11 * scale)
plt.yticks(fontsize=12 * scale)
plt.bar(X, Y)
plt.show()
