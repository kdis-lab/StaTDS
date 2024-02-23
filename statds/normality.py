import matplotlib.pyplot as plt
import numpy as np
import math

from . import stats
# ANALYTICAL METHODS


def shapiro_wilk_normality(data: np.array, alpha: float = 0.05):
    """
    Performs the Shapiro-Wilk test for normality. This test assesses the null hypothesis that the data
    was drawn from a normal distribution.
    
    Parameters
    ----------
    data : numpy.array
        Array of sample data. It should be a one-dimensional numpy array.
    alpha : float, optional
        The significance level for test analysis. Default is 0.05.
    
    Returns
    -------
    statistics_w : float
        The W statistic for the test, a measure of normality.
    p_value : float
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null hypothesis, indicating the data is not normally distributed.
    cv_value : float
        Critical value for the Shapiro-Wilk test. This is currently set to None.
    hypothesis : str
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null hypothesis can be rejected or not.

    """
    sorted_data = np.sort(data)

    mean = sorted_data.mean()
    sum_square = np.sum((np.array(sorted_data) - mean) ** 2)

    num_samples = data.shape[0]
    # m = num_samples // 2.0
    # Calculate Weights A
    a_weights = stats.get_shapiro_weights(num_samples)
    # Calculate b with = SUM_{i=1}^m a_i * (X_{n+1-i} - X_i{})
    # TODO CAMBIAR EL CALCULO DE B
    # https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    b = np.sum([a_weights[i] * (sorted_data[num_samples - (i + 1)] - sorted_data[i]) for i in range(len(a_weights))])

    statistics_w = (b ** 2) / sum_square

    # Calculate p-value with Shapiro-Wilk Tables
    # Preguntar esto:
    # https://sci2s.ugr.es/keel/pdf/algorithm/articulo/shapiro1965.pdf
    # https://real-statistics.com/tests-normality-and-symmetry/statistical-tests-normality-symmetry/shapiro-wilk-test/

    p_value, cv_value = stats.get_p_value_shapier(num_samples, statistics_w), None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
    return statistics_w, p_value, cv_value, hypothesis


def skewness(data: np.array, bias: bool = True):
    """
    Calculate the skewness of the given data. Skewness is a measure of the asymmetry of the probability distribution
    of a real-valued random variable about its mean. Positive skewness indicates a distribution with an asymmetric
    tail extending towards more positive values, while negative skewness indicates a distribution with an asymmetric
    tail extending towards more negative values.

    Parameters
    ----------
    data : numpy.array
        Array of sample data. It should be a one-dimensional numpy array.
    bias : bool, optional
        A boolean indicating whether to correct for statistical bias. If False, calculations are corrected for bias. 
        Default is True, meaning no bias correction is applied.

    Returns
    -------
    float
        The skewness of the data, indicating the degree of asymmetry in the distribution:
        - A value close to 0 indicates a symmetric distribution.
        - A positive value indicates a distribution that is skewed to the right.
        - A negative value indicates a distribution that is skewed to the left.

    Note
    ----
    The bias parameter affects the calculation of skewness. If bias is False, the result is adjusted for bias,
    making it suitable for samples from a larger population. If bias is True (default), the skewness of the sample
    is calculated without any adjustments.
    """
    mean = data.mean()
    x_minus_mean = data - mean
    num_samples = data.shape[0]
    m_3 = np.sum(x_minus_mean ** 3) / num_samples
    m_2 = np.sum(x_minus_mean ** 2) / num_samples
    statistical_skew = m_3 / (math.sqrt(m_2 ** 3))

    if bias is False:
        statistical_skew = (math.sqrt(num_samples * (num_samples - 1)) / (num_samples - 2)) * statistical_skew

    return statistical_skew


def skewness_test(b2, n):
    """
    Perform a skewness test to assess the asymmetry of the probability distribution of a dataset.
    This test calculates a z-score for the skewness and the corresponding p-value to determine the
    statistical significance of the skewness.

    Parameters
    ----------
    b2 : float
        The squared sample skewness.
    n : int
        The sample size.

    Returns
    -------
    z_b1 : float
        The z-score for the skewness test. A higher absolute value indicates greater skewness.
    p_value : float
        The p-value corresponding to the z-score. A small p-value (typically ≤ 0.05) suggests
        that the distribution of the data is significantly skewed.

    Note
    ----
    The skewness test is a hypothesis test that evaluates whether the skewness of the data differs
    significantly from 0, which would indicate a symmetric distribution. The test uses the sample size and
    the squared skewness to compute the test statistic. It's particularly useful for determining if a
    distribution departs from normality, where skewness is expected to be about 0.
    """
    y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) /
             ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)))
    w_2 = -1 + math.sqrt(2 * (beta2 - 1))
    # delta = 1 / math.sqrt(math.log(w_2))
    delta = 1 / math.sqrt(0.5 * math.log(w_2))
    alpha = math.sqrt(2.0 / (w_2 - 1))
    y = np.where(y == 0, 1, y)
    z_b1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))

    p_value = 2 * stats.get_p_value_normal(z_b1)
    return z_b1, p_value


def kurtosis(data: np.array, bias: bool = True):
    """
    Calculate the kurtosis of the given data. Kurtosis is a measure of the "tailedness" of the probability
    distribution of a real-valued random variable. In a general sense, kurtosis quantifies whether the tails
    of the data distribution are heavier or lighter compared to a normal distribution.

    Parameters
    ----------
    data : numpy.array
        Array of sample data. It should be a one-dimensional numpy array.
    bias : bool, optional
        A boolean indicating whether to correct for statistical bias. If False, calculations are corrected for bias. 
        Default is True, meaning no bias correction is applied.

    Returns
    -------
    float
        The kurtosis of the data, representing the degree of kurtosis in the distribution:
        - A value greater than 3 indicates a distribution with heavier tails and a sharper peak compared to a normal distribution.
        - A value less than 3 indicates a distribution with lighter tails and a flatter peak compared to a normal distribution.
        - A value close to 3, especially when bias is False, indicates a distribution similar to a normal distribution in terms of kurtosis.

    Note
    ----
    The bias parameter affects the calculation of kurtosis. If bias is False, the result is adjusted for bias,
    making it more representative for samples from a larger population. If bias is True (default), the kurtosis
    is calculated without any adjustments. The adjustment for bias involves a correction factor based on the number
    of samples, which corrects the result for the tendency of small samples to underestimate kurtosis.
    """
    mean = data.mean()
    x_minus_mean = data - mean
    num_samples = data.shape[0]
    m_4 = np.sum(x_minus_mean ** 4) / num_samples
    m_2 = np.sum(x_minus_mean ** 2) / num_samples
    statistical_kurtosis = m_4 / (m_2 ** 2)

    if bias is False:
        adj = ((num_samples - 1) / ((num_samples-2) * (num_samples-3)))
        statistical_kurtosis = adj * ((num_samples + 1) * statistical_kurtosis + 6)

    return statistical_kurtosis


def kurtosis_test(b2: float, n: int):
    """
    Perform a kurtosis test to assess the 'tailedness' or peakedness of the probability distribution of a dataset.
    This test calculates a z-score for the kurtosis and the corresponding p-value to determine the statistical
    significance of the kurtosis.

    Parameters
    ----------
    b2 : float
        The squared sample kurtosis.
    n : int
        The sample size.

    Returns
    -------
    z_b2 : float
        The z-score for the kurtosis test. This score indicates the degree of peakedness in the distribution.
    p_value : float
        The p-value corresponding to the z-score. A small p-value (typically ≤ 0.05) suggests that the
        distribution of the data has significant kurtosis, differing from a normal distribution.

    Note
    ----
    The kurtosis test is used to determine if a dataset has heavier or lighter tails compared to a normal
    distribution. The test involves calculating the mean and variance of b2, the standardized version of b2, and then
    computing the standardized moment of b2. The z-score and p-value help assess whether the kurtosis of the dataset
    significantly deviates from that of a normal distribution, where the kurtosis is expected to be around 0.
    """

    e_b2 = (3 * (n - 1)) / (n + 1)
    var_b2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    x = (b2 - e_b2) / math.sqrt(var_b2)

    beta2 = ((6 * (n ** 2 - 5 * n + 2)) / ((n + 7) * (n + 9))) * math.sqrt(
        (6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    a = 6 + 8.0 / beta2 * (2.0 / beta2 + math.sqrt((1 + (4.0 / (beta2 ** 2)))))
    z_b2 = ((1 - (2.0 / (9 * a))) - ((1 - (2 / a)) / (1 + x * math.sqrt(2.0 / (a - 4)))) ** (1 / 3)) / math.sqrt(
        2 / (9 * a))
    p_value = 2 * stats.get_p_value_normal(z_b2)

    return z_b2, p_value


def d_agostino_pearson(data: np.array, alpha: float = 0.05):
    """
    Perform the D'Agostino and Pearson's omnibus test for normality. This test combines skew and kurtosis
    to produce an omnibus test of normality, testing the null hypothesis that a sample comes from a normally
    distributed population.

    Parameters
    ----------
    data : numpy.array
        Array of sample data. It should be a one-dimensional numpy array.
    alpha : float, optional
        The significance level for test analysis. Default is 0.05.

    Returns
    -------
    statistic_dp : float
        The D'Agostino and Pearson's test statistic.
    p_value : float
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null hypothesis,
        indicating the data is not normally distributed.
    cv_value : float
        Critical value for the test based on the chi-squared distribution with 2 degrees of freedom.
    hypothesis : str
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null hypothesis can be rejected or not.

    Note
    ----
    The test calculates skewness and kurtosis of the data, standardizes these values, and then
    calculates a test statistic that follows a chi-squared distribution with 2 degrees of freedom under
    the null hypothesis. The p-value is then derived from this chi-squared distribution.
    """
    sorted_data = np.sort(data)

    skew = skewness(sorted_data)
    kurt = kurtosis(sorted_data)
    num_samples = sorted_data.shape[0]

    z_sqrt_b_1, _ = skewness_test(skew, num_samples)

    z_b_2, _ = kurtosis_test(kurt, num_samples)

    statistic_dp = z_sqrt_b_1**2 + z_b_2**2

    # Calculate p-value with chi^2 with 2 degrees of freedom
    p_value, cv_value = stats.get_p_value_chi2(statistic_dp, 2, alpha=alpha)
    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value >= alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return statistic_dp, p_value, cv_value, hypothesis


def kolmogorov_smirnov(data: np.array, alpha: float = 0.05):
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit. This non-parametric test compares a sample
    with a reference probability distribution (in this case, the normal distribution), assessing whether
    the sample data follows the same distribution as the reference distribution.

    Parameters
    ----------
    data : numpy.array
        Array of sample data. It should be a one-dimensional numpy array.
    alpha : float, optional
        The significance level for the test. Default is 0.05.

    Returns
    -------
    d_max : float
        The maximum difference between the Empirical Cumulative Distribution Function (ECDF) of the data
        and the Cumulative Distribution Function (CDF) of the reference distribution (normal distribution in this case).
    p_value : float
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null hypothesis,
        suggesting that the sample data does not follow the reference distribution.
    cv_value : float
        Critical value for the Kolmogorov-Smirnov test. This is currently set to None.
    hypothesis : str
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null hypothesis can be rejected or not.

    Note
    ----
    The test calculates the maximum difference (d_max) between the ECDF of the sample data and the CDF
    of the normal distribution. The KS statistic is then derived from this difference and the sample size. The
    p-value is estimated from the KS statistic. This test is non-parametric and does not assume a normal
    distribution of the data.
    """

    sorted_data = np.sort(data)

    # Calcular la función de distribución acumulativa empírica (ECDF)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    # Calcular la diferencia máxima entre la ECDF y la CDF de una distribución normal

    norm_cdf = [stats.get_cdf_normal((i - np.mean(sorted_data)) / (np.std(sorted_data))) for i in sorted_data]

    d_max = np.max(np.abs(ecdf - norm_cdf))

    # Calcular el estadístico de Kolmogorov-Smirnov (KS)
    ks_statistic = d_max * np.sqrt(n)

    # https://radzion.com/blog/probability/kolmogorov
    p_value = 1.0 - ks_statistic * (0.0498673470 - 0.142088994 + 0.0776645018 /
                                    (ks_statistic - 0.0122854966 + 0.253199760))

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
    cv_value = None

    return ks_statistic, p_value, cv_value, hypothesis

# GRAPHICAL METHODS


def qq_plot(data: np.array):
    """
    Generate a Quantile-Quantile (QQ) plot for a given dataset. A QQ plot is a graphical tool to help assess if a
    dataset follows a particular distribution, such as a normal distribution. It compares the quantiles of the data
    to the quantiles of a theoretical distribution.

    Parameters
    ----------
    data : numpy.array
        A numpy array of sample data to be analyzed.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object that contains the QQ plot.

    Note
    ----
    In the QQ plot, the quantiles of the data are plotted against the quantiles of a theoretical distribution
    (in this case, the normal distribution). If the data follow the theoretical distribution, the points on the QQ plot
    will approximately lie on the line y = x. Departures from this line indicate departures from the theoretical
    distribution. The plot helps in visually assessing normality, which is a common assumption in many statistical
    tests.
    """
    theoretical_quantiles = np.sort(np.random.normal(size=len(data)))
    data_quantiles = np.sort(data)

    plt.figure(figsize=(8, 8), facecolor='none')
    plt.plot(theoretical_quantiles, data_quantiles, 'o')
    plt.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
             [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
    plt.xlabel('Theorical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('QQ Plot')
    plt.grid(True)

    return plt.gcf()


def pp_plot(data: np.array):
    """
    Generate a Probability-Probability (PP) plot for a given dataset. A PP plot is a graphical technique for assessing
    how closely a set of data follows a given distribution, typically the normal distribution. It compares the empirical
    cumulative distribution function (CDF) of the data to the theoretical CDF of the specified distribution.

    Parameters
    ----------
    data : numpy.array
        A numpy array of sample data to be analyzed.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object that contains the PP plot.

    Note
    ----
    In a PP plot, the y-axis shows the empirical CDF of the data, and the x-axis displays the theoretical CDF of
    the distribution being tested (normal distribution in this case). A 45-degree reference line is added to the plot.
    If the data follow the specified distribution, the plot will align closely with this line. The PP plot is useful for
    assessing the goodness of fit of a distribution and is particularly helpful for determining if data follow a normal
    distribution, which is a common assumption in various statistical tests.
    """

    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    theoretical_cdf = [stats.get_cdf_normal(value) for value in sorted_data]

    plt.figure(figsize=(8, 8), facecolor='none')
    plt.scatter(theoretical_cdf, empirical_cdf)
    plt.plot([0, 1], [0, 1], color='red')  # Línea de referencia
    plt.xlabel('Empirical CDF')
    plt.ylabel('Theoretical CDF (Normal)')
    plt.title('PP Plot')
    plt.grid(True)

    return plt.gcf()
