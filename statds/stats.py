import math
import pandas as pd
import numpy as np
from pathlib import Path

current_directory = Path(__file__).resolve().parent


def get_cdf_normal(z_value):
    """
    Calculate the cumulative distribution function (CDF) of the standard normal distribution.

    Parameters
    ----------
    z_value : float
        The z-value for which the CDF is calculated.

    Returns
    -------
    float
        The CDF value for the given z-value.
    """
    return 0.5 * (1 + math.erf(z_value / math.sqrt(2)))


def get_p_value_normal(z_value):
    """
    Calculate the p-value for a given z-value in a standard normal distribution.

    Parameters
    ----------
    z_value : float
        The z-value for which the p-value is calculated.

    Returns
    -------
    float
        The p-value associated with the given z-value.
    """
    p_value = 1 - get_cdf_normal(abs(z_value))

    return p_value


def inverse_gaussian_cdf(value, mu: float = 1.0, lamb: float = 1.0):
    """
    Calculate the cumulative distribution function (CDF) of the inverse Gaussian distribution.

    Parameters
    ----------
    value : float
        The value for which the CDF is calculated.
    mu : float, optional
        The mean parameter of the distribution. Default is 1.0.
    lamb : float, optional
        The lambda parameter of the distribution. Default is 1.0.

    Returns
    -------
    float
        The CDF of the inverse Gaussian distribution for the given value.
    """
    a = math.sqrt(lamb / value) * (value / mu - 1)
    b = - math.sqrt(lamb / value) * (value / mu + 1)
    a = get_cdf_normal(a)
    b = get_cdf_normal(b)
    return a + math.exp(2 * lamb / mu) * b


def get_p_value_binomial(num: int, statistical: int, probability: float = 0.5):
    """
    Calculate the p-value of a binomial distribution.

    Parameters
    ----------
    num : int
        The number of trials in the binomial distribution.
    statistical : int
        The number of successful trials.
    probability : float, optional
        The probability of success in each trial. Default is 0.5.

    Returns
    -------
    float
        The p-value associated with the given values.
    """
    # Calculate the p-value of binomial distribution
    def binom_prob(n, k, p):
        return math.comb(n, k) * p ** k * (1 - p) ** (n - k)

    p_value = sum(binom_prob(num, i, probability) for i in range(0, num + 1) if binom_prob(num, i, probability) <=
                  binom_prob(num, statistical, probability))

    return p_value


def binomial_coef(n: int, k: int):
    """
    Calculate the binomial coefficient (n choose k).

    Parameters
    ----------
    n : int
        The total number of items.
    k : int
        The number of items to be chosen.

    Returns
    -------
    int
        The binomial coefficient value (n choose k).
    """
    if n < 0:
        return math.nan

    if n == 0 or k < 0 or n - k + 1 == 0:
        return 0

    return math.gamma(n + 1) / (math.gamma(k + 1) * math.gamma(n - k + 1))


def get_p_value_chi2(z_value: float, k_degrees_of_freedom: int, alpha: float):
    """
    Calculate the p-value of a binomial distribution.

    Parameters
    ----------
    z_value : float
        The z-value for which the CDF is calculated.
    k_degrees_of_freedom : int
        The degrees of freedom of the chi-squared distribution.
    alpha : float
        The significance level for which the critical value is needed.

    Returns
    -------
    p_value : float
        The p-value associated with the given z-value.
    cv_to_alpha : float
        The critical value corresponding to the specified alpha.
    """

    def chi_sq(z: float, k_freedom: int):
        """
        Calculate the chi-squared value for a given z-value and degrees of freedom.

        Parameters
        ----------
        z : float
            The z-value for which the chi-squared value is calculated.
        k_freedom : int
            The degrees of freedom for the chi-squared calculation.

        Returns
        -------
        float
            The chi-squared value.
        """
        if k_freedom == 1 and z > 1000:
            return 0
        if z > 1000 or k_freedom > 1000:
            q = chi_sq((z - k_freedom) ** 2 / (2 * k_freedom), 1) / 2
            return q if z > k_freedom else 1 - q

        p = math.exp(-0.5 * z)
        if k_freedom % 2 == 1:
            p *= math.sqrt(2 * z / math.pi)

        k = k_freedom
        while k >= 2:
            p *= z / k
            k -= 2

        t = p
        a = k_freedom
        while t > 1e-10 * p:
            a += 2
            t *= z / a
            p += t

        return 1 - p

    chi_table = pd.read_csv(current_directory / "assets/statistical_tables/chi_table.csv")

    columns = list(chi_table.columns)
    available_df = chi_table.DF.unique()
    available_alpha = [float(i) for i in columns[1:]]
    selected_alpha = min(available_alpha, key=lambda num: abs(float(num) - float(alpha)))
    selected_df = min(available_df, key=lambda num: abs(num - k_degrees_of_freedom))
    cv_to_alpha = float(chi_table[chi_table.DF == selected_df][str(selected_alpha)].iloc[0])

    p_value = chi_sq(z_value, k_degrees_of_freedom)

    return p_value, cv_to_alpha


def get_cv_q_distribution(k_degrees_of_freedom: int, num_alg: int, alpha: float):
    """
    Retrieve the critical value from a Q-table for a given chi-squared distribution. This function reads a Q-table
    from a CSV file, which contains critical values for various degrees of freedom, numerator degrees of freedom,
    and numbers of algorithms. It selects the critical value based on the provided degrees of freedom 
    (k_degrees_of_freedom), number of algorithms (num_alg), and significance level (alpha).

    Parameters
    ----------
    k_degrees_of_freedom : int
        The degrees of freedom of the chi-squared distribution.
    num_alg : int
        The number of algorithms.
    alpha : float
        The significance level for which the critical value is needed.

    Returns
    -------
    float
        The critical value corresponding to the input degrees of freedom, number of algorithms, and alpha.
    """

    table_of_q = pd.read_csv(current_directory / "assets/statistical_tables/CV_q_table.csv")
    columns = list(table_of_q.columns)

    available_df = table_of_q[columns[0]].unique()
    available_df_numerator = table_of_q[columns[1]].unique()
    available_num_alg = [int(i) for i in columns[2:]]
    selected_alpha = min(available_df_numerator, key=lambda num: abs(num - alpha))
    selected_df = min(available_df, key=lambda num: abs(num - k_degrees_of_freedom))
    selected_num_alg = min(available_num_alg, key=lambda num: abs(num - num_alg))

    critical_value = float(table_of_q[(table_of_q[columns[0]] == selected_df) &
                                      (table_of_q[columns[1]] == selected_alpha)][str(selected_num_alg)].iloc[0])

    return critical_value


def get_q_alpha_nemenyi(num_algorithm: int, alpha: float):
    """
    Retrieve the Q_alpha value from a Nemenyi critical values table for a given number of algorithms and alpha.
    This function reads a Nemenyi critical values table from a CSV file, which contains critical values for
    different numbers of algorithms and significance levels (alpha). It selects the Q_alpha value based on the
    provided number of algorithms (num_algorithm) and significance level (alpha).

    Parameters
    ----------
    num_algorithm : int
        The number of algorithms for which the Q_alpha value is needed.
    alpha : float
        The significance level for which the Q_alpha value is needed.

    Returns
    -------
    float
        The Q_alpha value corresponding to the input number of algorithms and alpha.
    """
    nemenyi_table = pd.read_csv(current_directory / "assets/statistical_tables/nemenyi_table.csv")

    q_alpha = float(nemenyi_table[nemenyi_table['models'] == num_algorithm][str(alpha)].iloc[0])

    return q_alpha


def get_cv_f_distribution(df_numerator: int, df_denominator: int, alpha: float):
    """
    Retrieve the critical value from an F-distribution critical values table for given degrees of freedom and alpha.
    This function reads an F-distribution critical values table from a CSV file, which contains critical values for
    different degrees of freedom for the numerator and denominator, as well as significance levels (alpha). It
    selects the critical value based on the provided degrees of freedom for the numerator (df_numerator), degrees
    of freedom for the denominator (df_denominator), and significance level (alpha).

    Parameters
    ----------
    df_numerator : int
        Degrees of freedom for the numerator.
    df_denominator : int
        Degrees of freedom for the denominator.
    alpha : float
        The significance level for which the critical value is needed.

    Returns
    -------
    float
        The critical value corresponding to the input degrees of freedom and alpha.
    """
    cv_table = pd.read_csv(current_directory / "assets/statistical_tables/f_table.csv")
    columns = list(cv_table.columns)
    available_df_denominator = cv_table[columns[0]].unique()
    available_df_numerator = [int(i) for i in columns[2:]]
    selected_df_denominator = min(available_df_denominator, key=lambda num: abs(num - df_denominator))
    selected_df_numerator = min(available_df_numerator, key=lambda num: abs(num - df_numerator))

    crit_value = float(cv_table[(cv_table[columns[0]] == selected_df_denominator) &
                                (cv_table[columns[1]] == alpha)][str(selected_df_numerator)].iloc[0])

    return crit_value


def get_cv_t_distribution(k_degrees_of_freedom: int, alpha: float):
    """
    Retrieve the critical value from a t-distribution critical values table for a given degrees of freedom and
    alpha. This function reads a t-distribution critical values table from a CSV file, which contains critical
    values for different degrees of freedom and significance levels (alpha). It selects the critical value based on
    the provided degrees of freedom (k_degrees_of_freedom) and significance level (alpha).

    Parameters
    ----------
    k_degrees_of_freedom : int
        Degrees of freedom for the t-distribution.
    alpha : float
        The significance level for which the critical value is needed.

    Returns
    -------
    float
        The critical value corresponding to the input degrees of freedom and alpha.
    """
    cv_table = pd.read_csv(current_directory / "assets/statistical_tables/t_table.csv")
    columns = list(cv_table.columns)

    available_df = cv_table[columns[0]].unique()
    available_df_numerator = [float(i) for i in columns[1:]]
    selected_alpha = min(available_df_numerator, key=lambda num: abs(num - alpha))
    selected_df = min(available_df, key=lambda num: abs(num - k_degrees_of_freedom))
    crit_value = float(cv_table[cv_table[columns[0]] == selected_df][str(selected_alpha)].iloc[0])

    return crit_value


def get_cv_willcoxon(num_problems: int, alpha: float):
    """
    Retrieve the critical value from a Wilcoxon signed-rank test critical values table for a given number of
    samples and alpha. This function reads a Wilcoxon signed-rank test critical values table from a CSV file,
    which contains critical values for different numbers of samples and significance levels (alpha). It selects the
    critical value based on the provided number of samples (num_problems) and significance level (alpha).

    Parameters
    ----------
    num_problems : int
        The number of samples for which the critical value is needed.
    alpha : float
        The significance level for which the critical value is needed.

    Returns
    -------
    int
        The critical value corresponding to the input number of samples and alpha.
    """
    wilcoxon_table = pd.read_csv(current_directory / "assets/statistical_tables/CV_Wilcoxon.csv")

    columns = wilcoxon_table.columns
    available_num_samples = wilcoxon_table[columns[0]].unique()
    available_alpha = [float(i) for i in columns[1:]]
    selected_alpha = min(available_alpha, key=lambda num: abs(num - alpha))
    selected_num_samples = min(available_num_samples, key=lambda num: abs(num - num_problems))

    cv_alpha_selected = wilcoxon_table[wilcoxon_table[columns[0]] == selected_num_samples][str(selected_alpha)].iloc[0]
    try:
        cv_alpha_selected = int(cv_alpha_selected)
    except ValueError:
        cv_alpha_selected = None
    return cv_alpha_selected


def get_shapiro_weights(n_weights: int):
    """
    Retri: inteve weights a_i for any given sample size n_weights.

    Parameters
    ----------
    n_weights : int
        The number of samples.

    Returns
    -------
    list
        Weights a_i list for the given number of samples.
    """
    table = pd.read_csv(current_directory / "assets/statistical_tables/shapiro_weights.csv")
    row_table = table[table["n"] == n_weights].to_numpy()
    first_nan_index = np.argmax(np.isnan(row_table))
    weights = row_table[0][1:first_nan_index]
    return weights


def get_p_value_shapier(num_samples: int, statistics_w: float):
    """
    Retrieve the p-value from a Shapiro-Wilk test for a given number of samples and statistics.

    Parameters
    ----------
    num_samples : int
        The number of samples for which the critical value is needed.
    statistics_w : float
        The statistics for which the p-value is calculated.

    Returns
    -------
    float
        The p-value from the Shapiro-Wilk test.
    """

    df = pd.read_csv(current_directory / "assets/statistical_tables/shapiro_table.csv")
    selected_num_samples = min(df["n"].to_numpy(), key=lambda num: abs(num - num_samples))
    columns_table = list(df.columns)[1:]
    selected_row = df[df["n"] == selected_num_samples][columns_table].to_numpy()[0]
    aux = [abs(i - statistics_w) for i in selected_row]
    index_min = aux.index(min(aux))
    p_value_1 = float(columns_table[index_min])
    statistics_w_min_1 = selected_row[index_min]
    aux.pop(index_min)
    index_min = aux.index(min(aux))
    p_value_2 = float(columns_table[index_min+1])
    statistics_w_min_2 = selected_row[index_min+1]

    m = (p_value_2 - p_value_1) / (statistics_w_min_2 - statistics_w_min_1)

    p_value = abs(p_value_1 + m * (statistics_w - statistics_w_min_1))

    return p_value


def get_p_value_f(value: float, df_numerator: int, df_denominator: int):
    """
    Calculate the p-value for a given F-statistic using the F-distribution. This function is used to determine the
    significance of a test statistic from an ANOVA test, comparing the ratio of variances between and within groups.

    Parameters
    ----------
    value : float
        The F-statistic value for which the p-value is to be calculated.
    df_numerator : int
        The degrees of freedom for the numerator, often corresponding to the number of groups minus one.
    df_denominator : int
        The degrees of freedom for the denominator, usually related to the total number of observations minus the number
        of groups.

    Returns
    -------
    float
        The p-value corresponding to the given F-statistic and degrees of freedom. This p-value indicates the
        probability of observing a value at least as extreme as the F-statistic under the null hypothesis.

    Note
    ----
    The calculation of the p-value depends on the degrees of freedom for both the numerator and the denominator, and the
    F-statistic itself.
    """
    def stat_com(q: float, i: int, j: int, b: float):
        """
        Calculate a statistical component used in the computation of p-values for certain distributions,
        such as the F-distribution. This function is a helper function and is typically used within other
        statistical functions to simplify their calculations.

        Parameters
        ----------
        q : float
            A parameter typically representing a ratio of variance or a transformed probability value.
        i : int
            The starting value for a series in the computation, usually relating to degrees of freedom or similar
            metrics.
        j : int
            The ending value for the series in the computation.
        b : float
            A base value used in the calculation, often related to degrees of freedom or other distribution parameters.

        Returns
        -------
        float
            The calculated statistical component based on the input parameters. This value contributes to the overall
            calculation of a p-value or other statistical measures in larger functions.
        """
        zz = 1
        z = zz
        aux = i
        while aux <= j:
            zz *= q * aux / (aux - b)
            z += zz
            aux += 2
        return z
    x = df_denominator / (df_numerator * value + df_denominator)
    if df_numerator % 2 == 0:
        return (stat_com(1 - x, df_denominator, df_numerator + df_denominator - 4, df_denominator - 2) *
                math.pow(x, df_denominator / 2.0))
    if df_denominator % 2 == 0:
        return (1 - stat_com(x, df_numerator, df_numerator + df_denominator - 4, df_numerator - 2) *
                math.pow(1 - x, df_numerator / 2.0))

    th = math.atan(math.sqrt(df_numerator * value / df_denominator))
    a = th / (math.pi / 2.0)
    sth = math.sin(th)
    cth = math.cos(th)
    if df_denominator > 1:
        a += sth * cth * stat_com(cth ** 2, 2, df_denominator - 3, -1) / (math.pi / 2.0)
    if df_numerator == 1:
        return 1 - a

    c = 4 * stat_com(sth ** 2, df_denominator + 1, df_numerator + df_denominator - 4,
                     df_denominator - 2) * sth * math.pow(cth, df_denominator) / math.pi
    if df_denominator == 1:
        return 1 - a + c / 2.0

    k = 2
    while k <= (df_denominator - 1) / 2.0:
        c *= k / (k - 0.5)
        k += 1
    return abs(1 - a + c)


def get_pdf_t(z_value: float, k_degrees_of_freedom: int):
    """
    Calculate the t value for a given z-value and degrees of freedom.

    Parameters
    ----------
    z_value : float
        The z-value for which the chi-squared value is calculated.
    k_degrees_of_freedom : int
        The degrees of freedom for the chi-squared calculation.

    Returns
    -------
    float
        The t value.
    """

    numerator = math.gamma((k_degrees_of_freedom + 1) / 2)
    denominator = np.sqrt(k_degrees_of_freedom * np.pi) * math.gamma(k_degrees_of_freedom / 2)
    b = ((1 + (z_value**2) / k_degrees_of_freedom) ** (- (k_degrees_of_freedom + 1) / 2))
    result = (numerator / denominator) * b
    return result


def get_p_value_t(z_value: float, k_degrees_of_freedom: int, inf_approx=100, steps=1000000):
    """
    Calculate the p-value associated with a given z-value and degrees of freedom using
    the t-distribution. This function approximates the p-value by converting the z-value
    to a t-value and then integrating over the t-distribution to find the area under the curve.

    Parameters
    ----------
    z_value : float
        The z-value for which the p-value is to be calculated.
    k_degrees_of_freedom : int
        The degrees of freedom for the t-distribution.
    inf_approx : int, optional
        The upper limit for the integral approximation, default is 100.
    steps : int, optional
        The number of steps to use in the numerical approximation, default is 1,000,000.

    Returns
    -------
    float
        The p-value corresponding to the given z-value and degrees of freedom.
    """
    h = (inf_approx - z_value) / steps
    sum_f = 0.5 * (get_pdf_t(z_value, k_degrees_of_freedom) + get_pdf_t(inf_approx, k_degrees_of_freedom))
    for i in range(1, steps):
        sum_f += get_pdf_t(z_value + i * h, k_degrees_of_freedom)
    return h * sum_f


def kolmogorov_p_value(d, n):
    """
    Calcula el p-valor para el estadístico de Kolmogorov-Smirnov.

    :param d: El estadístico de Kolmogorov-Smirnov.
    :param n: El número de observaciones.
    :return: El p-valor aproximado.
    """
    summa = 0
    for i in range(1, 1000000):  # Aumenta el rango para mayor precisión
        summa += (-1) ** (i - 1) * math.exp(-2 * i ** 2 * d ** 2 * n)
    p_value = 2 * summa
    return p_value
