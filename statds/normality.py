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
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null hypothesis, indicating
        the data is not normally distributed.
    cv_value : float
        Critical value for the Shapiro-Wilk test. This is currently set to None.
    hypothesis : str
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null
        hypothesis can be rejected or not.

    """
    data = np.ravel(data).astype(np.float64)
    n = len(data)
    a = np.zeros(n, dtype=np.float64)

    # Sort the data and center it by subtracting the median
    y = np.sort(data)
    y -= data[n // 2]  # Subtract the median (or a nearby value)
    statistics_w, pw, _ = _swilk(y, a[:n // 2], False)

    p_value, cv_value = pw, None

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if p_value > alpha:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha}(Same distributions)"
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
        A value greater than 3 indicates a distribution with heavier tails and a sharper peak compared to a normal
        distribution.
        A value less than 3 indicates a distribution with lighter tails and a flatter peak compared to a normal
        distribution.
        A value close to 3, especially when bias is False, indicates a distribution similar to a normal distribution
        in terms of kurtosis.

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
        adj = ((num_samples - 1) / ((num_samples - 2) * (num_samples - 3)))
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
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null
        hypothesis can be rejected or not.

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

    statistic_dp = z_sqrt_b_1 ** 2 + z_b_2 ** 2

    # Calculate p-value with chi^2 with 2 degrees of freedom
    p_value, cv_value = stats.get_p_value_chi2(statistic_dp, 2, alpha=alpha)
    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if p_value > alpha:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha}(Same distributions)"

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
        A string stating the conclusion of the test based on the p-value and alpha. It indicates whether the null
        hypothesis can be rejected or not.

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

    # Calcular la diferencia máxima entre la ECDF y la CDF de una distribución normal
    # norm_cdf = [stats.get_cdf_normal((i - np.mean(sorted_data)) / (np.std(sorted_data))) for i in sorted_data]
    norm_cdf = [stats.get_cdf_normal(i) for i in sorted_data]

    dplus = (np.arange(1.0, n + 1) / n - norm_cdf)
    amax = dplus.argmax()
    dplus = dplus[amax]
    dminus = (norm_cdf - np.arange(0.0, n) / n)
    amax = dminus.argmax()
    dminus = dminus[amax]
    d_max = dminus if dminus > dplus else dplus
    # Calcular el estadístico de Kolmogorov-Smirnov (KS)
    ks_statistic = d_max

    p_value = stats.kolmogorov_p_value(ks_statistic, n)

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if p_value > alpha:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha}(Same distributions)"
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


def _ppnd(p):
    split = 0.42

    a0 = 2.50662823884
    a1 = -18.61500062529
    a2 = 41.39119773534
    a3 = -25.44106049637
    b1 = -8.47351093090
    b2 = 23.08336743743
    b3 = -21.06224101826
    b4 = 3.13082909833
    c0 = -2.78718931138
    c1 = -2.29796479134
    c2 = 4.85014127135
    c3 = 2.32121276858
    d1 = 3.54388924762
    d2 = 1.63706781897

    q = p - 0.5
    if abs(q) <= split:
        r = q * q
        temp = q * (((a3 * r + a2) * r + a1) * r + a0)
        temp = temp / ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.)
        return temp  # , 0

    r = p
    if q > 0:
        r = 1 - p
    if r > 0:
        r = math.sqrt(-math.log(r))
    else:
        return 0.  # , 1

    temp = (((c3 * r + c2) * r + c1) * r + c0)
    temp /= (d2 * r + d1) * r + 1.
    return -temp if q < 0 else temp  # , 0


def _poly(c, nord, x):
    res = c[0]
    if nord == 1:
        return res

    p = x * c[nord - 1]
    if nord == 2:
        return res + p

    for ind in range(nord - 2, 0, -1):
        p = (p + c[ind]) * x
    res += p
    return res


def _alnorm(x, upper):
    """
    Helper function for swilk.

    Evaluates the tail area of the standardized normal curve from x to inf
    if upper is True or from -inf to x if upper is False

    Modification has been done to the Fortran version in November 2001 with the
    following note;

        MODIFY UTZERO.  ALTHOUGH NOT NECESSARY
        WHEN USING ALNORM FOR SIMPLY COMPUTING PERCENT POINTS,
        EXTENDING RANGE IS HELPFUL FOR USE WITH FUNCTIONS THAT
        USE ALNORM IN INTERMEDIATE COMPUTATIONS.

    The change is shown below as a commented utzero definition
    """
    ltone = 7.
    utzero = 38.
    con = 1.28

    a1 = 0.398942280444
    a2 = 0.399903438504
    a3 = 5.75885480458
    a4 = 29.8213557808
    a5 = 2.62433121679
    a6 = 48.6959930692
    a7 = 5.92885724438
    b1 = 0.398942280385
    b2 = 3.8052e-8
    b3 = 1.00000615302
    b4 = 3.98064794e-4
    b5 = 1.98615381364
    b6 = 0.151679116635
    b7 = 5.29330324926
    b8 = 4.8385912808
    b9 = 15.1508972451
    b10 = 0.742380924027
    b11 = 30.789933034
    b12 = 3.99019417011
    z = x

    if not (z > 0):  # negative of the condition to catch NaNs
        upper = False
        z = -z
    if not ((z <= ltone) or (upper and z <= utzero)):
        return 0. if upper else 1.
    y = 0.5 * z * z
    if z <= con:
        temp = 0.5 - z * (a1 - a2 * y / (y + a3 - a4 / (y + a5 + a6 / (y + a7))))
    else:
        temp = b1 * math.exp(-y) / (z - b2 + b3 / (z + b4 + b5 / (z - b6 + b7 /
                                                                  (z + b8 - b9 / (z + b10 + b11 / (z + b12))))))

    return temp if upper else (1 - temp)


def _swilk(x: np.array, a: np.array, init=False, n1=-1):
    """
    Calculates the Shapiro-Wilk W test and its significance level. This function is an adaptation
    from the original FORTRAN 77 code, with modifications for Python usage.

    The Shapiro-Wilk test is used to check the null hypothesis that a sample x comes from
    a normally distributed population. This function computes the test statistic (W) and its
    significance level (p-value), considering possible adjustments for small sample sizes.

    Parameters
    ----------
    x : list
        The sample data array, sorted in ascending order.
    a : list
        Coefficients for the Shapiro-Wilk W test statistic, typically precomputed
        for a given sample size.
    init : bool, optional
        A flag to indicate if the 'a' coefficients have already been initialized.
        Defaults to False, which means the coefficients will be initialized within the function.
    n1 : int, optional
        Adjusted sample size parameter, useful in case of censored data. Defaults to -1,
        which means no adjustment is made.

    Returns
    -------
    w : float
        The Shapiro-Wilk W test statistic.
    pw : float
        The p-value associated with the W test statistic. A small p-value suggests the
        sample is not normally distributed.
    ifault : int
        An error code (0 for no error; other values indicate different error conditions or warnings).

    Notes
    -----
    This implementation is a direct translation from the original algorithm published by
    Royston P., and it retains much of the original structure and variable names used in
    the FORTRAN code for ease of comparison and verification against the original.
    """
    n = len(x)
    n2 = len(a)
    upper = True
    c1 = [0., 0.221157, -0.147981, -0.207119e1, 0.4434685e1, -0.2706056e1]
    c2 = [0., 0.42981e-1, -0.293762, -0.1752461e1, 0.5682633e1, -0.3582633e1]
    c3 = [0.5440, -0.39978, 0.25054e-1, -0.6714e-3]
    c4 = [0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2]
    c5 = [-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2]
    c6 = [-0.4803, -0.82676e-1, 0.30302e-2]
    c7 = [0.164, 0.533]
    c8 = [0.1736, 0.315]
    c9 = [0.256, -0.635e-2]
    g = [-0.2273e1, 0.459]
    z90 = 0.12816e1
    z95 = 0.16449e1
    z99 = 0.23263e1
    zm = 0.17509e1
    zss = 0.56268
    bf1 = 0.8378
    xx90 = 0.556
    xx95 = 0.622
    sqrth = math.sqrt(2) / 2.0
    pi6 = 6 / np.pi
    small = 1e-19

    if n1 < 0:
        n1 = n
    nn2 = n // 2
    if nn2 < n2:
        return 1., 1., 3
    if n < 3:
        return 1., 1., 1
    w = 1.
    pw = 1.
    an = n

    if not init:
        if n == 3:
            a[0] = sqrth
        else:
            an25 = an + 0.25
            summ2 = 0.
            for ind1 in range(n2):
                temp = _ppnd((ind1 + 1 - 0.375) / an25)
                a[ind1] = temp
                summ2 += temp ** 2

            summ2 *= 2.
            ssumm2 = math.sqrt(summ2)
            rsn = 1 / math.sqrt(an)
            a1 = _poly(c1, 6, rsn) - (a[0] / ssumm2)
            if n > 5:
                i1 = 2
                a2 = -a[1] / ssumm2 + _poly(c2, 6, rsn)
                fac = math.sqrt((summ2 - (2 * a[0] ** 2) - 2 * a[1] ** 2) /
                                (1 - (2 * a1 ** 2) - 2 * a2 ** 2))
                a[1] = a2
            else:
                i1 = 1
                fac = math.sqrt((summ2 - 2 * a[0] ** 2) / (1 - 2 * a1 ** 2))

            a[0] = a1
            for ind1 in range(i1, nn2):
                a[ind1] *= -1. / fac
        init = True

    if n1 < 3:
        return w, pw, 1
    ncens = n - n1

    if ncens < 0 or ((ncens > 0) and (n < 20)):
        return w, pw, 4

    delta = ncens / an
    if delta > 0.8:
        return w, pw, 5

    range_1 = x[n1 - 1] - x[0]
    if range_1 < small:
        return w, pw, 6

    xx = x[0] / range_1
    sx = xx
    sa = -a[0]
    ind2 = n - 2
    for ind1 in range(1, n1):
        xi = x[ind1] / range_1
        sx += xi
        if ind1 != ind2:
            sa += (-1 if ind1 < ind2 else 1) * a[min(ind1, ind2)]
        xx = xi
        ind2 -= 1

    ifault = 0
    if n > 5000:
        ifault = 2

    sa /= n1
    sx /= n1
    ssa, ssx, sax = 0., 0., 0.
    ind2 = n - 1
    for ind1 in range(n1):
        if ind1 != ind2:
            asa = (-1 if ind1 < ind2 else 1) * a[min(ind1, ind2)] - sa
        else:
            asa = -sa

        xsx = x[ind1] / range_1 - sx
        ssa += asa * asa
        ssx += xsx * xsx
        sax += asa * xsx
        ind2 -= 1

    ssassx = math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)
    w = 1 - w1

    # Calculate significance level for W (exact for N=3)
    if n == 3:
        if w < 0.75:
            return 0.75, 0., ifault
        else:
            pw = 1. - pi6 * math.acos(math.sqrt(w))
            return w, pw, ifault

    y = math.log(w1)
    xx = math.log(an)
    if n <= 11:
        gamma = _poly(g, 2, an)
        if y >= gamma:
            return w, small, ifault
        y = -math.log(gamma - y)
        m = _poly(c3, 4, an)
        s = math.exp(_poly(c4, 4, an))
    else:
        m = _poly(c5, 4, xx)
        s = math.exp(_poly(c6, 3, xx))

    if ncens > 0:
        ld = -math.log(delta)
        bf = 1 + xx * bf1
        z90_f = z90 + bf * _poly(c7, 2, xx90 ** xx) ** ld
        z95_f = z95 + bf * _poly(c8, 2, xx95 ** xx) ** ld
        z99_f = z99 + bf * _poly(c9, 2, xx) ** ld
        zfm = (z90_f + z95_f + z99_f) / 3.
        zsd = (z90 * (z90_f - zfm) + z95 * (z95_f - zfm) + z99 * (z99_f - zfm)) / zss
        zbar = zfm - zsd * zm
        m += zbar * s
        s *= zsd

    pw = _alnorm((y - m) / s, upper)

    return w, pw, ifault
