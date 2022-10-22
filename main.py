<<<<<<< HEAD
import nltk
from nltk.corpus import brown
import pylab
import numpy as np

# load the 

# nltk.download("brown")
print(brown.categories())
# words = brown.words()
words = brown.words(fileids=['ca16'])
print(words)
# count words
fdist = nltk.FreqDist(w.lower() for w in words)
print(fdist)







# plot
# x = np.linspace(0, 20, 1000)  # 100 evenly-spaced values from 0 to 50
# y = np.sin(x)
# 
# pylab.plot(x, y)
# pylab.show()

=======
print("hello world!")
>>>>>>> 068a2a95baf618ee67a25c44995b8450cbc55b66
