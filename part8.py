from nltk.corpus import brown
from sklearn.utils import shuffle
import nltk
import numpy as np

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

# import nltk
# nltk.download('brown')
"""We want to repeat 6) by exploring the length of the sentences (in terms of number of tokens) in Brown corpus.
 Suggest a script that extracts sentences and calculate the length of sentences.
 Then draw the corresponding log-log plot and test the Zipf’s law distribution."""

# NOTE: I KNOW  I COULD HAVE DONE THIS BY MAKING A SOME FUNCTIONS AND NOT TO JUST COPY THE SAME CODE FOR EVERY SECTION
# i'M AWARE OF THAT THIS IS SPAGETTI CODE

# Split the brown corpus into 8 samples
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

# len(bronw.sents()) 57340

# ------- Taking the sentence out of the corpus -------

########### --- C1 --- ##########
# first initiolize
C1_sent = []
for i in id1:
    for j in brown.sents(fileids=i):
        C1_sent.append(len(j))

# len(C1_sent) 11656

# Get the freqvences of sentencess

freqC1 = nltk.FreqDist(w for w in C1_sent)
# FreqDist({5: 1116, 7: 992, 8: 868, 6: 744, 10: 744, 9: 620, 4: 620, 3: 620, 12: 496, 11: 434, ...}
# len 39

# Take  freqvencies from cormus C1
keysC1 = list(freqC1.keys())
fC1 = []

for i in keysC1:
    fC1.append(freqC1[i])

fC1.sort(reverse=True)

################## {C1, C2} ###############
C12_sent = []
for i in c12:
    for j in brown.sents(fileids=i):
        C12_sent.append(len(j))

# len(C12_sent)
# 14332

freqC12 = nltk.FreqDist(w for w in C12_sent)
# FreqDist({11: 566, 13: 546, 14: 541, 15: 531, 12: 527, 10: 525, 9: 518, 8: 495, 17: 482, 7: 471, ...})
# 97

# Take freqvencies from cormpus {C1, C2}

keysC12 = list(freqC12.keys())
fC12 = []

for i in keysC12:
    fC12.append(freqC12[i])
fC12.sort(reverse=True)

##################{C1, C2, C3, C4}################3

# ---- Select the subcorpus:
C1234_sent = []
# add the others
for i in c1234:
    for j in brown.sents(fileids=i):
        C1234_sent.append(len(j))

len(C1234_sent)  # 28383

# Get the freqvences of words
freqC1234 = nltk.FreqDist(w for w in C1234_sent)
# {11: 1071, 14: 1045, 9: 1020, 12: 1015, 13: 1010, 15: 1009, 10: 1007, 8: 954, 17: 944, 18: 931, ...}
# lne 112

# Take freqvencies from freqDis {C1, C2, C3, C4}
keysC1234 = list(freqC1234.keys())
fC1234 = []

for i in keysC1234:
    fC1234.append(freqC1234[i])

fC1234.sort(reverse=True)

############# {C1, C2, C3, C4, C5, C6, C7, C8}###########

# Get the freqvences of words: 1161192
All_sent = []
for i in c_all:
    for j in brown.sents(fileids=i):
        All_sent.append(len(j))

# len 57340

freqC_all = nltk.FreqDist(w for w in All_sent)
# FreqDist({11: 2142, 14: 2060, 13: 2041, 10: 2030, 9: 2015, 15: 2013, 12: 1975, 8: 1964, 16: 1958, 17: 1912, ...}
# len 126
# Take freqvencies from cormpus {all}

keysCall = list(freqC_all.keys())  # get keys
fCall = []  # make list for frecvencies

for i in keysCall:
    fCall.append(freqC_all[i])
fCall.sort(reverse=True)

######## .....  PLotting  .... ########
N1 = len(fC1)
N2 = len(fC12)
N3 = len(fC1234)
N4 = len(fCall)

# X-axel, 'rank'
rank1 = list(range(1, N1 + 1))
rank2 = list(range(1, N2 + 1))
rank3 = list(range(1, N3 + 1))
rank4 = list(range(1, N4 + 1))

# Make list into array
y1 = np.array(fC1)
y2 = np.array(fC12)
y3 = np.array(fC1234)
y4 = np.array(fCall)

# Frequency log
log_y1 = np.log(y1)
log_y2 = np.log(y2)
log_y3 = np.log(y3)
log_y4 = np.log(y4)

# rank long
log_rank1 = np.log(rank1)
log_rank2 = np.log(rank2)
log_rank3 = np.log(rank3)
log_rank4 = np.log(rank4)

# Plot 1
# plt.figure(figsize=(20, 20)) # to increase the plot resolution
# plt.ylabel("Log Frequency", fontsize=24)
# plt.xlabel("Log Ranks", fontsize=24)
# plt.suptitle("Fitting the Zipf's Law to sentence length", fontsize=24)
# plt.xticks(rotation=90) # to rotate x-axis values
# plt.scatter(log_rank1, log_y1)
#
##Plot 2
# plt.figure(figsize=(20, 20)) # to increase the plot resolution
# plt.ylabel("Log Frequency", fontsize=24)
# plt.xlabel("Log Ranks", fontsize=24)
# plt.suptitle("Fitting the Zipf's Law to sentence length", fontsize=24)
# plt.xticks(rotation=90) # to rotate x-axis values
# plt.scatter(log_rank2, log_y2)
#
##Plot 3
# plt.figure(figsize=(20, 20)) # to increase the plot resolution
# plt.ylabel("Log Frequency", fontsize=24)
# plt.xlabel("Log Ranks", fontsize=24)
# plt.suptitle("Fitting the Zipf's Law to sentence length", fontsize=24)
# plt.xticks(rotation=90) # to rotate x-axis values
# plt.scatter(log_rank3, log_y3)
#
##Plot 4
# plt.figure(figsize=(20, 20)) # to increase the plot resolution
# plt.ylabel("Log Frequency", fontsize=24)
# plt.xlabel("Log Ranks", fontsize=24)
# plt.suptitle("Fitting the Zipf's Law to sentence length", fontsize=24)
# plt.xticks(rotation=90) # to rotate x-axis values
# plt.scatter(log_rank, log_y4)


# ------ TESTING -----


# corrrelation

correlation1 = pearsonr(log_rank1, log_y1)
correlation2 = pearsonr(log_rank2, log_y2)
correlation3 = pearsonr(log_rank3, log_y3)
correlation4 = pearsonr(log_rank4, log_y4)
print(correlation1, correlation2, correlation3, correlation4)

# ---- Getting intervals ----


def get_prediction_interval(prediction, y_test, test_predictions, pi=0.95):

    """Get a prediction interval for a linear regression.
    INPUTS:
    - Single prediction,
    - y_test- All test set predictions,
    - Prediction interval threshold (default = .95)
    OUTPUT:
    - Prediction interval for single prediction"""

    # get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions) ** 2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)

    # get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev

    # generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    return lower, prediction, upper


# ------------ c1 - model ----------
# create 2d array from x-variable
x_log_reshaped = log_rank1.reshape(-1, 1)
# fit the model
model = LinearRegression().fit(x_log_reshaped, log_y1)
# coefficient of determination
y_pred = model.predict(x_log_reshaped)
model_line = y_pred

# print(f"predicted response:\n{y_pred}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")

# 95 % confidence intervals
## ----------Plot and save confidence interval of linear regression - 95%

lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper = get_prediction_interval(i, log_y1, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)

# Draw plot with 95% CI : dataset C1
plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.fill_between(
    log_rank1, upper_vet, lower_vet, color="b", label="Confidence Interval"
)
plt.scatter(log_rank1, log_y1, color="orange", label="Real data")
plt.plot(log_rank1, model_line, "k", label="Linear regression")
plt.title("95% confidence interval", fontsize=24)
plt.ylabel("Log Frequency", fontsize=24)
plt.xlabel("Log Ranks", fontsize=24)
plt.legend()
plt.show()

# ------------ {C1,C2} model ---------

# create 2d array from x-variable
x_log_reshaped = log_rank2.reshape(-1, 1)
# fit the model
model = LinearRegression().fit(x_log_reshaped, log_y2)
# coefficient of determination
y_pred = model.predict(x_log_reshaped)
model_line = y_pred

# print(f"predicted response:\n{y_pred}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")

# 95 % confidence intervals
## ----------Plot and save confidence interval of linear regression - 95%

lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper = get_prediction_interval(i, log_y2, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)

# Draw plot with 95% CI : dataset C1,C2
plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.fill_between(
    log_rank2, upper_vet, lower_vet, color="b", label="Confidence Interval"
)
plt.scatter(log_rank2, log_y2, color="orange", label="Real data")
plt.plot(log_rank2, model_line, "k", label="Linear regression")
plt.title("95% confidence interval", fontsize=24)
plt.ylabel("Log Frequency", fontsize=24)
plt.xlabel("Log Ranks", fontsize=24)
plt.legend()
plt.show()

# ------------ {C1,C2,C3,C4} model ---------

# create 2d array from x-variable
x_log_reshaped = log_rank3.reshape(-1, 1)
# fit the model
model = LinearRegression().fit(x_log_reshaped, log_y3)
# coeffiecient of determination
y_pred = model.predict(x_log_reshaped)
model_line = y_pred

# print(f"predicted response:\n{y_pred}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")

# 95 % confidence intervals
## ----------Plot and save confidence interval of linear regression - 95%

lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper = get_prediction_interval(i, log_y3, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)

# Draw plot with 95% CI : dataset C1,C2
plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.fill_between(
    log_rank3, upper_vet, lower_vet, color="b", label="Confidence Interval"
)
plt.scatter(log_rank3, log_y3, color="orange", label="Real data")
plt.plot(log_rank3, model_line, "k", label="Linear regression")
plt.title("95% confidence interval", fontsize=24)
plt.ylabel("Log Frequency", fontsize=24)
plt.xlabel("Log Ranks", fontsize=24)
plt.legend()
plt.show()

# ------------ {all} model ---------

# create 2d array from x-variable
x_log_reshaped = log_rank4.reshape(-1, 1)
# fit the model
model = LinearRegression().fit(x_log_reshaped, log_y4)
# coeffiecient of determination
y_pred = model.predict(x_log_reshaped)
model_line = y_pred

# print(f"predicted response:\n{y_pred}")
# print(f"intercept: {model.intercept_}")
# print(f"slope: {model.coef_}")

# 95 % confidence intervals
## ----------Plot and save confidence interval of linear regression - 95%

lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper = get_prediction_interval(i, log_y4, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)

# Draw plot with 95% CI : dataset C1,C2
plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.fill_between(
    log_rank4, upper_vet, lower_vet, color="b", label="Confidence Interval"
)
plt.scatter(log_rank4, log_y4, color="orange", label="Real data")
plt.plot(log_rank4, model_line, "k", label="Linear regression")
plt.title("95% confidence interval", fontsize=24)
plt.ylabel("Log Frequency", fontsize=24)
plt.xlabel("Log Ranks", fontsize=24)
plt.legend()
plt.show()

# ----- HOW MANY POINTS OUT OF THE CONFIDENCE BAR ------

# C1:0.02564102564102564
1 / N1

# C1,C2: 0.020618556701030927
2 / N2

# C1,C2,C3,C4: 0.026785714285714284
3 / N3

# All: 0.023809523809523808
3 / N4
