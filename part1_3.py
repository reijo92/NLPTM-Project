import nltk
import numpy as np
import matplotlib.pyplot as plt


# from nltk.corpus import stopwords
from nltk.book import FreqDist

from sklearn.linear_model import LinearRegression
from scipy import stats
from tabulate import tabulate

# import statistics


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


def plot_bar(collection, scale=1):

    plt.figure(figsize=(6 * scale, 12 * scale))
    plt.ylabel("Frequency", fontsize=12 * scale)
    plt.xlabel("Word", fontsize=12 * scale)
    plt.xticks(rotation=90)  # to rotate x-axis values
    plt.xticks(fontsize=12 * scale)
    plt.yticks(fontsize=12 * scale)
    for i in range(30):
        plt.bar(collection[i][0], collection[i][1])
    plt.show()


def plot_zipfs(
    x_log,
    y_log,
    scale=1,
    linreg=False,
    confint=False,
    model_line=None,
    upper_vet=None,
    lower_vet=None,
):

    plt.figure(figsize=(6 * scale, 12 * scale), dpi=80)
    plt.ylabel("Log Frequency", fontsize=12 * scale)
    plt.xlabel("Log Ranks", fontsize=12 * scale)
    plt.xticks(fontsize=12 * scale)
    plt.yticks(fontsize=12 * scale)
    # plt.title("Bigrams", fontsize=12 * scale)
    if linreg:
        if confint:
            plt.fill_between(
                x_log, upper_vet, lower_vet, color="b", label="Confidence Interval"
            )
            plt.plot(x_log, y_log, color="orange", label="Real data")
            plt.plot(x_log, model_line, "k", label="Linear regression")
            plt.legend(loc=1, prop={"size": 10 * scale})
        else:
            plt.plot(x_log, model_line)
            plt.plot(x_log, y_log)
    else:
        plt.plot(x_log, y_log)
    plt.show()


def calc_confint_bounds(y, model_line):
    """Plot and save confidence interval of linear regression  - 95%""" 
    lower_vet = []
    upper_vet = []
    ten, twentyfive, fifty, seventyfive = True, True, True, True

    for i in model_line:
        lower, prediction, upper = get_prediction_interval(i, y, model_line)
        lower_vet.append(lower)
        upper_vet.append(upper)
        ready = round(len(lower_vet) / len(model_line), 3)
        if ten and ready == 0.1:
            print("10 % done calculating interval.")
            ten = False
        elif twentyfive and ready == 0.25:
            print("25 % done calculating interval.")
            twentyfive = False
        elif fifty and ready == 0.5:
            print("50 % done calculating interval.")
            fifty = False
        elif seventyfive and ready == 0.75:
            print("75 % done calculating interval.")
            seventyfive = False

    lower_vet = np.array(lower_vet)
    upper_vet = np.array(upper_vet)

    return lower_vet, upper_vet, prediction


def calc_correlation(x, y):
    correlation = np.corrcoef(x, y)[0, 1]
    return correlation


def fit_model(x, y):
    # create 2d array from x-variable
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression().fit(x_reshaped, y)
    y_pred = model.predict(x_reshaped)
    model_line = y_pred
    return model_line


# def remove_stopwords():
#     get common english stopwords
#     stop_words = set(stopwords.words("english"))
#     words_cleaned = [word for word in words_lower if word not in stop_words]


def calc_freq(words):
    freq = FreqDist(words)
    N = len(freq)
    freq = freq.most_common()  # ordering
    # list containing number of occasions
    num_occasion = [freq[i][1] for i in range(N)]
    num_occasion = np.array(num_occasion)
    return freq, num_occasion, N


def estimate_words_out(y, lower, upper):
    out_words = []
    for i, freq in enumerate(y):
        if freq < lower[i]:
            out_words.append(freq)
        elif freq > upper[i]:
            out_words.append(freq)
    return out_words


def calc_table(mydata):
    head = ["Frequency", "Correlation", "Words Out (%)"]
    print(tabulate(mydata, headers=head, tablefmt="grid"))


if __name__ == "__main__":

    brown = nltk.corpus.brown
    words = brown.words()  # words in whole corpus
    words_lower = [word.lower() for word in words if word.isalpha()]

    exercise = input("Choose exercise 1-3: ")
    if exercise == "1":
        print("Analysing unigrams...")
        # %% Converto lower case
        freq_list, y, N = calc_freq(words_lower)
    elif exercise == "2":
        print("Analysing bigrams...")
        bigrams = nltk.bigrams(words_lower)
        freq_list, y, N = calc_freq(bigrams)
    elif exercise == "3":
        print("Analysing trigrams...")
        trigrams = nltk.trigrams(words_lower)
        freq_list, y, N = calc_freq(trigrams)

    x = [(j + 1) for j in range(N)]  # list of ranks

    # change to logrithmic scale
    x_log = np.log(x)
    y_log = np.log(y)

    if exercise == "1":
        plot_bar(freq_list, scale=1)

    # plot zipfs law log-scale
    plot_zipfs(x_log, y_log, scale=3)
    # estimate model
    model_line = fit_model(x_log, y_log)
    # fit linear regression
    plot_zipfs(x_log, y_log, linreg=True, model_line=model_line, scale=3)
    # calculate correlation
    correlation = calc_correlation(x_log, y_log)
    print("Calculating confidence interval.")
    print("This could take some time...")
    # calculate confidence intervals
    lower, upper, prediction = calc_confint_bounds(y_log, model_line)
    # draw linear regression with confidence bounds
    plot_zipfs(
        x_log,
        y_log,
        linreg=True,
        confint=True,
        model_line=model_line,
        lower_vet=lower,
        upper_vet=upper,
        scale=3,
    )
    # estimate words that are outside of 95% confidence interval
    out_words = estimate_words_out(y_log, lower, upper)
    perc_out = len(out_words) / len(y_log)
    if exercise == "1" or exercise == "2":
        print(
            f"Percentage of tokens that do not fall within 95 % bound: {perc_out*100} %"
        )
        print(f"Estimation of the linear curve parameter (correlation): {correlation}")
    # summarise results to a table in exercise 3
    if exercise == "3":

        print("Starting to make summary data frame...")
        corr_tri = correlation
        perc_out_tri = perc_out
        N_tri = N

        freq_list_uni, y_uni, N_uni = calc_freq(words_lower)
        bigrams = nltk.bigrams(words_lower)
        freq_list_bi, y_bi, N_bi = calc_freq(bigrams)

        x_uni = [(j + 1) for j in range(N_uni)]  # list of ranks
        x_bi = [(j + 1) for j in range(N_bi)]  # list of ranks

        # change to logrithmic scale
        x_log_uni = np.log(x_uni)
        y_log_uni = np.log(y_uni)
        x_log_bi = np.log(x_bi)
        y_log_bi = np.log(y_bi)

        # calculate model, confidence intervals and out words
        # for unigrams
        model_line_uni = fit_model(x_log_uni, y_log_uni)
        corr_uni = calc_correlation(x_log_uni, y_log_uni)
        lower_uni, upper_uni, prediction_uni = calc_confint_bounds(
            y_log_uni, model_line_uni
        )

        out_words_uni = estimate_words_out(y_log_uni, lower_uni, upper_uni)
        perc_out_uni = len(out_words_uni) / len(y_log_uni)

        # calculate model, confidence intervals and out words
        # for bigrams
        model_line_bi = fit_model(x_log_bi, y_log_bi)
        corr_bi = calc_correlation(x_log_bi, y_log_bi)
        lower_bi, upper_bi, prediction_bi = calc_confint_bounds(y_log_bi, model_line_bi)
        out_words_bi = estimate_words_out(y_log_bi, lower_bi, upper_bi)
        perc_out_bi = len(out_words_bi) / len(y_log_bi)

        data = [
            ["Unigrams", N_uni, corr_uni, perc_out_uni],
            ["Bigrams", N_bi, corr_bi, perc_out_bi],
            ["Trigrams", N_tri, corr_tri, perc_out_tri],
        ]
        calc_table(data)
