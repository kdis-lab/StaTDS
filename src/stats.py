import math
import pandas as pd
import numpy as np
from pathlib import Path

current_directory = Path(__file__).resolve().parent


def get_cdf_normal(z_value):
    """
        Calculate the cumulative distribution function (CDF) of the standard normal distribution.
    :param z_value: The z-value for which the CDF is calculated.
    :return: The CDF value for the given z-value.
    """
    return 0.5 * (1 + math.erf(z_value / math.sqrt(2)))


def get_p_value_normal(z_value):
    """
        Calculate the p-value for a given z-value in a standard normal distribution.
    :param z_value: The z-value for which the p-value is calculated.
    :return: The p-value associated with the given z-value.
    """
    p_value = 1 - get_cdf_normal(z_value)

    return p_value


def inverse_gaussian_cdf(value, mu: float = 1.0, lambda_: float = 1.0):
    """
        Calculate the cumulative distribution function (CDF) of the inverse Gaussian distribution.
    :param value: The value for which the CDF is calculated.
    :param mu: The mean parameter of the distribution (default is 1.0).
    :param lambda_: The lambda parameter of the distribution (default is 1.0).
    :return:
    """
    a = math.sqrt(lambda_ / value) * (value / mu - 1)
    b = - math.sqrt(lambda_ / value) * (value / mu + 1)
    a = get_cdf_normal(a)
    b = get_cdf_normal(b)
    return a + math.exp(2 * lambda_ / mu) * b


def get_p_value_binomial(num: int, statistical: int, probability: float = 0.5):
    """
        Calculate the p-value of a binomial distribution.
    :param num: The number of trials in the binomial distribution.
    :param statistical: The number of successful trials.
    :param probability: The probability of success in each trial (default is 0.5).
    :return:
    """
    # Calculate the p-value of binomial distribution
    comb = math.comb(num, statistical)

    p_value = comb * pow(probability, statistical) * pow(1 - probability, num - statistical)

    return p_value


def binomial_coef(n: int, k: int):
    """
        Calculate the binomial coefficient (n choose k).
    :param n: The total number of items.
    :param k: The number of items to be chosen.
    :return: The binomial coefficient value (n choose k).
    """
    if n < 0:
        return math.nan

    if n == 0 or k < 0 or n - k + 1 == 0:
        return 0 

    a = math.gamma(n + 1) / (math.gamma(k + 1) * math.gamma(n - k + 1))

    return math.gamma(n + 1) / (math.gamma(k + 1) * math.gamma(n - k + 1))


def get_p_value_chi2(z_value: float, k_degrees_of_freedom: int, alpha: float):
    """
        Calculate the p-value of a binomial distribution.
    :param z_value: The z-value for which the CDF is calculated.
    :param k_degrees_of_freedom: The degrees of freedom of the chi-squared distribution.
    :param alpha:
    :return: p_value and cv_to_alpha
    """
    def calcular_pdf_chi2(value, df):
        """
            Calculates the Probability Density Function (PDF) of the chi-squared distribution.
        :param value: The value for which the PDF will be calculated.
        :param df: The degrees of freedom of the chi-squared distribution.
        :return: The PDF value for the given value and degrees of freedom.
        """
        return (1 / (2 ** (df / 2) * math.gamma(df / 2))) * value ** (df / 2 - 1) * math.exp(-value / 2)

    def calcular_cdf_chi2(value, degrees_of_freedom, step=0.0001):
        """
        Calculates the Cumulative Distribution Function (CDF) of the chi-squared distribution.

        :param value: The value for which the CDF will be calculated.
        :param degrees_of_freedom: The degrees of freedom of the chi-squared distribution.
        :param step: The step size for numerical approximation. Smaller step size results in higher precision.
        :return: The CDF value for the given value and degrees of freedom.
        """
        cdf = 0
        x = 0
        while x <= value:
            cdf += calcular_pdf_chi2(x, degrees_of_freedom) * step
            x += step
        return cdf

    chi_table = pd.read_csv(current_directory / "assets/statistical_tables/chi_table.csv")

    columns = list(chi_table.columns)
    available_df = chi_table.DF.unique()
    available_alpha = [float(i) for i in columns[1:]]
    selected_alpha = min(available_alpha, key=lambda num: abs(float(num) - float(alpha)))
    selected_df = min(available_df, key=lambda num: abs(num - k_degrees_of_freedom))
    cv_to_alpha = float(chi_table[chi_table.DF == selected_df][str(selected_alpha)].iloc[0])

    p_value = 1 - calcular_cdf_chi2(z_value, k_degrees_of_freedom)

    return p_value, cv_to_alpha


def get_cv_q_distribution(k_degrees_of_freedom: int, num_alg: int, alpha: float):
    """
        Retrieve the critical value from a Q-table for a given chi-squared distribution. This function reads a Q-table
        from a CSV file, which contains critical values for various degrees of freedom,
        numerator degrees of freedom, and numbers of algorithms. It selects the critical value based on the provided
        degrees of freedom (k_degrees_of_freedom), number of algorithms (num_alg), and significance level (alpha).
    :param k_degrees_of_freedom: The degrees of freedom of the chi-squared distribution.
    :param num_alg: The number of algorithms.
    :param alpha: The significance level for which the critical value is needed.
    :return: The critical value corresponding to the input degrees of freedom, number of algorithms, and alpha.
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
    :param num_algorithm: The number of algorithms for which the Q_alpha value is needed.
    :param alpha: The significance level for which the Q_alpha value is needed.
    :return: The Q_alpha value corresponding to the input number of algorithms and alpha.
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
    :param df_numerator: Degrees of freedom for the numerator.
    :param df_denominator: Degrees of freedom for the denominator.
    :param alpha: The significance level for which the critical value is needed.
    :return: The critical value corresponding to the input degrees of freedom and alpha.
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
    :param k_degrees_of_freedom: Degrees of freedom for the t-distribution.
    :param alpha: The significance level for which the critical value is needed.
    :return: The critical value corresponding to the input degrees of freedom and alpha.
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
    :param num_problems: The number of samples for which the critical value is needed.
    :param alpha: The significance level for which the critical value is needed.
    :return: The critical value (integer) corresponding to the input number of samples and alpha.
    """
    wilcoxon_table = pd.read_csv(current_directory / "assets/statistical_tables/CV_Wilcoxon.csv")

    columns = wilcoxon_table.columns
    available_num_samples = wilcoxon_table[columns[0]].unique()
    available_alpha = [float(i) for i in columns[1:]]
    selected_alpha = min(available_alpha, key=lambda num: abs(num - alpha))
    selected_num_samples = min(available_num_samples, key=lambda num: abs(num - num_problems))

    cv_alpha_selected = wilcoxon_table[wilcoxon_table[columns[0]] == selected_num_samples][str(selected_alpha)].iloc[0]

    return int(cv_alpha_selected)


def get_shapiro_weights(n_weights):
    table = pd.read_csv(current_directory / "assets/statistical_tables/shapiro_weights.csv")
    row_table = table[table["n"] == n_weights].to_numpy()
    first_nan_index = np.argmax(np.isnan(row_table))
    weights = row_table[0][1:first_nan_index]
    return weights


def get_p_value_shapier(num_samples: int, statistics_w: float):
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

    p_value = p_value_1 + m * (statistics_w - statistics_w_min_1)

    return p_value
