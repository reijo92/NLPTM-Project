from nltk.book import *
from nltk.corpus import brown

import csv
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS


sents = brown.sents()  # sentences of the corpus
sentsA = [s for s in sents if len(s) < 5]  # subset containing only short sentences
sentsB = [s for s in sents if len(s) > 9]  # subset containinn only long sentences


# writing sentsA to a file
myFile = open("short_sent.csv", "w")
writer = csv.writer(myFile)
for data_list in sentsA:
    writer.writerow(data_list)
myFile.close()

# writing sentsB to a file
myFile = open("long_sent.csv", "w")
writer = csv.writer(myFile)
for data_list in sentsB:
    writer.writerow(data_list)
myFile.close()


# function drawing the WordCloud
def DrawCloud(list):
    sanat = ""
    sanat += " ".join(list) + " "
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stopwords,
        min_font_size=10,
    ).generate(sanat)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# end of the function


# reading the sentences from files to lists

sentsA = []
sentsB = []
with open("short_sent.csv", "r") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        sentsA.append(row)
csvfile.close()

with open("long_sent.csv", "r") as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        sentsB.append(row)
csvfile.close()


"""
transforming the two lists of sentences to two lists of words containing only
purely alphabetic words as lower case strings
"""

NA = len(sentsA)  # Number of short sentences
NB = len(sentsB)  # Number of long sentences

sentsA2 = []
sentsB2 = []

for i in range(NA):  # going through all sentences of the set
    MA = len(sentsA[i])
    for j in range(MA):  # going through all words of the sentence
        w = sentsA[i][j]
        if w.isalpha():  # if purely alphabetic,
            sentsA2.append(w.lower())  # added to list in lower case format

for i in range(NB):  # the same as above
    MB = len(sentsB[i])
    for j in range(MB):
        w = sentsB[i][j]
        if w.isalpha():
            sentsB2.append(w.lower())


# drawing Wordclouds for both sets

if len(sentsA2) > 0:
    DrawCloud(sentsA2)  # short sentences
else:
    print("No short sentences")

if len(sentsB2) > 0:
    DrawCloud(sentsB2)  # long sentences
else:
    print("No long sentences")
