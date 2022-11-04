import nltk
from nltk.book import *
from nltk.corpus import brown

from matplotlib import pyplot as plt
import numpy as np

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy import stats


sanat = brown.words()  # words of corpus
sanat2 = [w for w in sanat if w.isalpha()]  # only words that contain only letters


# creating part-of-speech-tags
tagged = nltk.pos_tag(sanat2)


# isolating the tag part
tags = [tagged[i][1] for i in range(len(sanat2))]


# frequency distribution of tags
fdist = FreqDist(tags)
y = fdist.most_common()  # ordering
N = len(sanat2)  # number of words
N2 = len(y)  # number of tags
y2 = [y[i][1] for i in range(N2)]  # list containing number of occasions
x = [(j + 1) for j in range(N2)]  # list of ranks


y3 = np.array(y2)  # list to array
y4 = y3 * (1 / N)  # numbers to proportions


y5 = np.log(y4)  # log of frequence

x2 = np.log(x)  # log of rank

# %% Python visualization with pyplot
scale = 2
plt.figure(figsize=(6 * scale, 12 * scale), dpi=80)
plt.ylabel("Log Frequency", fontsize=12 * scale)
plt.xlabel("Log Ranks", fontsize=12 * scale)
plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)
# plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.suptitle("Fitting the Zipf's Law", fontsize=24)
# plt.xticks(rotation=90)  # to rotate x-axis values

x_log = x2
y_log = y5
plt.scatter(x_log, y_log)
plt.show()


# linear regression
# Dependent variable y = frequency
# Independent variable x = rank


# corrrelation
correlation = pearsonr(x_log, y_log)
print(correlation)

# create 2d array from x-variable
x_log_reshaped = x_log.reshape(-1, 1)

# fit the model
model = LinearRegression().fit(x_log_reshaped, y_log)
# coeffiecient of determination
# r_sq = model.score(x_log, y_log)
y_pred = model.predict(x_log_reshaped)
model_line = y_pred

print(f"predicted response:\n{y_pred}")
# print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# %% Python visualization with pyplot
plt.figure(figsize=(6 * scale, 12 * scale), dpi=80)
plt.title("Fitting the Zipf's Law", fontsize=24)
plt.ylabel("Log Frequency", fontsize=12 * scale)
plt.xlabel("Log Ranks", fontsize=12 * scale)
# plt.xticks(rotation=90)  # to rotate x-axis values
plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)
# plt.plot(y_log)
plt.plot(x_log, model_line)
plt.scatter(x_log, y_log)
plt.show()


# 95 % confidence intervals


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


## Plot and save confidence interval of linear regression  - 95%
lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper = get_prediction_interval(i, y_log, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)


plt.figure(figsize=(6 * scale, 12 * scale), dpi=80)
# plt.figure(figsize=(20, 20))  # to increase the plot resolution
plt.fill_between(x_log, upper_vet, lower_vet, color="b", label="Confidence Interval")
plt.scatter(x_log, y_log, color="orange", label="Real data")
plt.plot(x_log, model_line, "k", label="Linear regression")
plt.xticks(fontsize=12 * scale)
plt.yticks(fontsize=12 * scale)
plt.ylabel("Log Frequency", fontsize=12 * scale)
plt.xlabel("Log Ranks", fontsize=12 * scale)
plt.title("95% confidence interval", fontsize=24)
plt.legend(loc=1, prop={"size": 10 * scale})
# plt.xlim()
# plt.ylim(-1000,8000)
plt.show()


print(y_log)

out_words = list()
for i, freq in enumerate(y_log):
    if freq < lower_vet[i]:
        out_words.append(freq)
    elif freq > upper_vet[i]:
        out_words.append(freq)


percentage_out = len(out_words) / len(y_log)
print(percentage_out)
print(len(out_words))
print(len(y_log))
