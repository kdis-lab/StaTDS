import numpy as np
import math
import pandas as pd
import stats


def shapiro_wilk_normality(data: np.array, alpha: float = 0.05):
    """
        Perform the Shapiro-Wilk test for normality. The Shapiro-Wilk test tests the null hypothesis that the data was
        drawn from a normal distribution.

        :param data: Array of sample data.
        :param alpha: The significance level for test analysis.

        :return: A tuple containing the following:
                - statistics_w: The W statistic for the test, a measure of normality.
                - p_value: The p-value for the hypothesis test. A small p-value (typically ≤ 0.05)
                  rejects the null hypothesis, indicating the data is not normally distributed.
                - cv_value: Critical value for the Shapiro-Wilk test. This is currently set to None.
                - hypothesis: A string stating the conclusion of the test based on the p-value and alpha.
                  It indicates whether the null hypothesis can be rejected or not.
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

        :param data: Array of sample data. It should be a one-dimensional numpy array.
        :param bias: A boolean (True or False). If False, then the calculations are corrected for statistical bias.

        :return: The skewness of the data. It is a float representing the degree of skewness in the distribution:
            - A value close to 0 indicates a symmetric distribution.
            - A positive value indicates a distribution that is skewed to the right.
            - A negative value indicates a distribution that is skewed to the left.

        Note: The bias parameter affects the calculation of skewness. If bias is False, the result is adjusted for bias,
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


def calculate_z_b_1(b2, n):  # TODO Pensar si merece la pena cambiar el nombre a skewness_test y que devuelva el p_valor
    y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = (3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) /
             ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)))
    w_2 = -1 + math.sqrt(2 * (beta2 - 1))
    # delta = 1 / math.sqrt(math.log(w_2))
    delta = 1 / math.sqrt(0.5 * math.log(w_2))
    alpha = math.sqrt(2.0 / (w_2 - 1))
    y = np.where(y == 0, 1, y)
    z_b1 = delta * np.log(y / alpha + np.sqrt((y / alpha) ** 2 + 1))
    return z_b1


def kurtosis(data: np.array, bias: bool = True):
    """
        Calculate the kurtosis of the given data. Kurtosis is a measure of the "tailedness" of the probability
        distribution of a real-valued random variable. In a general sense, kurtosis quantifies whether the tails
        of the data distribution are heavier or lighter compared to a normal distribution.

        :param data: Array of sample data. It should be a one-dimensional numpy array.
        :param bias: A boolean (True or False). If False, then the calculations are corrected for statistical bias.

        :return: The kurtosis of the data. It is a float representing the degree of kurtosis in the distribution:
            - A value greater than 3 indicates a distribution with heavier tails and a sharper peak compared to a normal
              distribution.
            - A value less than 3 indicates a distribution with lighter tails and a flatter peak compared to a normal
              distribution.
            - A value close to 3, especially when bias is False, indicates a distribution similar to a normal
              distribution in terms of kurtosis.

        Note: The bias parameter affects the calculation of kurtosis. If bias is False, the result is adjusted for bias,
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


def calculate_z_b_2(b2, n):  # TODO Pensar si merece la pena cambiar el nombre a kurtosis_test y que devuelva el p_valor
    # Step 2: Calculate the mean and variance of b2
    e_b2 = (3 * (n - 1)) / (n + 1)

    var_b2 = (24 * n * (n - 2) * (n - 3)) / ((n + 1) ** 2 * (n + 3) * (n + 5))

    # Step 3: Compute the standardized version of b2
    x = (b2 - e_b2) / math.sqrt(var_b2)

    # Step 4: Compute the third standardized moment of b2

    beta2 = ((6 * (n ** 2 - 5 * n + 2)) / ((n + 7) * (n + 9))) * math.sqrt(
        (6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))

    # Step 5: Compute A
    a = 6 + 8.0 / beta2 * (2.0 / beta2 + math.sqrt((1 + (4.0 / (beta2 ** 2)))))

    # Step 6: Compute Z(b2)
    z_b2 = ((1 - (2.0 / (9 * a))) - ((1 - (2 / a)) / (1 + x * math.sqrt(2.0 / (a - 4)))) ** (1 / 3)) / math.sqrt(
        2 / (9 * a))

    return z_b2


def d_agostino_pearson(data: np.array, alpha: float = 0.05):
    """
        Perform the D'Agostino and Pearson's omnibus test for normality. This test combines skew and kurtosis
        to produce an omnibus test of normality, testing the null hypothesis that a sample comes from a normally
        distributed population.

        :param data: Array of sample data. It should be a one-dimensional numpy array.
        :param alpha: The significance level for test analysis, default is 0.05.

        :return: A tuple containing the following:
            - statistic_dp: The D'Agostino and Pearson's test statistic.
            - p_value: The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null
              hypothesis,
              indicating the data is not normally distributed.
            - cv_value: Critical value for the test based on the chi-squared distribution with 2 degrees of freedom.
            - hypothesis: A string stating the conclusion of the test based on the p-value and alpha.
              It indicates whether the null hypothesis can be rejected or not.

        Note: The test calculates skewness and kurtosis of the data, standardizes these values, and then
        calculates a test statistic that follows a chi-squared distribution with 2 degrees of freedom under
        the null hypothesis. The p-value is then derived from this chi-squared distribution.
    """
    sorted_data = np.sort(data)

    skew = skewness(sorted_data)
    kurt = kurtosis(sorted_data)
    num_samples = sorted_data.shape[0]

    z_sqrt_b_1 = calculate_z_b_1(skew, num_samples)

    z_b_2 = calculate_z_b_2(kurt, num_samples)

    statistic_dp = z_sqrt_b_1**2 + z_b_2**2

    # Calculate p-value with chi^2 with 2 degrees of freedom
    p_value, cv_value = stats.get_p_value_chi2(statistic_dp, 2, alpha=alpha)

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return statistic_dp, p_value, cv_value, hypothesis


def kolmogorov_smirnov(data: np.array, alpha: float = 0.05):
    """
        Perform the Kolmogorov-Smirnov test for goodness of fit. This non-parametric test compares a sample
        with a reference probability distribution (in this case, the normal distribution), assessing whether
        the sample data follows the same distribution as the reference distribution.

        :param data: Array of sample data. It should be a one-dimensional numpy array.
        :param alpha: The significance level for the test, default is 0.05.

        :return: A tuple containing the following:
            - d_max: The maximum difference between the Empirical Cumulative Distribution Function (ECDF) of the data
              and the Cumulative Distribution Function (CDF) of the reference distribution (normal distribution in this
              case).
            - p_value: The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null
              hypothesis,
              suggesting that the sample data does not follow the reference distribution.
            - cv_value: Critical value for the Kolmogorov-Smirnov test. This is currently set to None.
            - hypothesis: A string stating the conclusion of the test based on the p-value and alpha.
              It indicates whether the null hypothesis can be rejected or not.

        Note: The test calculates the maximum difference (d_max) between the ECDF of the sample data and the CDF
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
    # TODO ARREGLAR ESTO ESTA MAL
    # https://radzion.com/blog/probability/kolmogorov
    p_value = 1.0 - ks_statistic * (0.0498673470 - 0.142088994 + 0.0776645018 /
                                    (ks_statistic - 0.0122854966 + 0.253199760))

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
    cv_value = None

    return ks_statistic, p_value, cv_value, hypothesis


def levene_test(dataset: pd.DataFrame, alpha: float = 0.05, center: str = 'mean'):
    """
        Perform the Levene test for equality of variances. This test is used to assess whether the variances of two
        or more groups are equal. It is an essential test before conducting ANOVA, as ANOVA assumes homogeneity of
        variances.

        :param dataset: A pandas DataFrame where each column represents a different group/sample. The first column is
        ignored.
        :param alpha: The significance level for the test, default is 0.05.
        :param center: The method for calculating the center of each group - 'mean', 'median', or 'trimmed'. Default is
        'mean'.

        :return: A tuple containing the following:
            - statistic_levene: The Levene test statistic. A higher value indicates a greater likelihood of differing
              variances.
            - p_value: The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null
              hypothesis, suggesting that the variances across groups are not equal.
            - rejected_value: The critical value for the test at the specified alpha level.
            - hypothesis: A string stating the conclusion of the test based on the test statistic and alpha.
              It indicates whether the null hypothesis of equal variances can be rejected or not.

        Note: The Levene test is robust to non-normal distributions, making it preferable to Bartlett's test when data
        are not normally distributed. The choice of 'center' parameter (mean, median, trimmed) can affect the test's
        sensitivity to departures from normality. The test statistic is computed based on the absolute deviations from
        the group centers and then compared against an F-distribution to obtain the p-value.
    """

    def trimmed_mean(data, proportion_to_cut=0.1):
        n = len(data)
        if n == 0:
            return float('nan')  # Return NaN if data is empty
        cut_count = int(proportion_to_cut * n)
        trimmed_data = sorted(data)[cut_count:-cut_count]

        return sum(trimmed_data) / len(trimmed_data)

    if dataset.shape[1] < 2:
        raise "Error: Levene Test need at least two samples"

    if center not in ["mean", "median", "trimmed"]:
        raise "Error: Center of Levene Test must be 'mean', 'median' or 'trimmed'"

    names_groups = list(dataset.columns)[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)

    if center == "mean":
        calculate_center = np.mean
    elif center == "median":
        calculate_center = np.median
    else:
        calculate_center = trimmed_mean

    center_of_groups = [calculate_center(dataset[i]) for i in names_groups]

    z_ij = [abs(dataset[names_groups[i]] - center_of_groups[i]).to_numpy() for i in range(len(names_groups))]

    z_bar_i = np.array([np.mean(i) for i in z_ij])
    z_bar = [i * j for i, j in zip(z_bar_i, num_samples)]
    z_bar = np.sum(z_bar) / num_total

    numer = (num_total - len(names_groups)) * np.sum(num_samples * (z_bar_i - z_bar) ** 2)
    dvar = np.sum([np.sum((z_ij[i] - z_bar_i[i]) ** 2, axis=0) for i in range(num_groups)])

    denom = (len(names_groups) - 1) * dvar

    statistic_levene = (numer / denom)

    # TODO Calculate P-Valor F Distribution with alpha, k-1, N-k (Revisar tras hablar)
    # p_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    rejected_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistic_levene < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    # print("\t p_value = ", p_value)

    return statistic_levene, p_value, rejected_value, hypothesis


def bartlett_test(dataset: pd.DataFrame, alpha: float = 0.05):
    """
        Perform Bartlett's test for homogeneity of variances. This test checks the null hypothesis that all input
        samples (represented as columns in the dataset) come from populations with equal variances. It is commonly used
        before conducting ANOVA, as equal variances are an assumption of ANOVA.

        :param dataset: A pandas DataFrame where each column represents a different group/sample. The first column is
        ignored.
        :param alpha: The significance level for the test, default is 0.05.

        :return: A tuple containing the following:
            - statistical_bartlett: Bartlett's test statistic. A higher value indicates a greater likelihood of
              differing variances.
            - p_value: The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) rejects the null
              hypothesis,
              suggesting that the variances across groups are not equal.
            - cv_value: Critical value for the test at the specified alpha level.
            - hypothesis: A string stating the conclusion of the test based on the test statistic and alpha.
              It indicates whether the null hypothesis of equal variances can be rejected or not.

        Note: Bartlett's test is sensitive to departures from normality. Therefore, if the data are not normally
        distributed, Levene's test is a more appropriate choice. The test statistic is compared against a chi-squared
        distribution to obtain the p-value. The formula for the test statistic takes into account the number of groups
        and the total number of samples.
    """
    if dataset.shape[1] < 2:
        raise "Error: Bartlett Test need at least two samples"

    names_groups = list(dataset.columns)[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)

    std_groups = [np.std(dataset[i]) for i in names_groups]

    pooled_variance = (np.sum([((num_samples[i] - 1) * (std_groups[i] ** 2)) for i in range(num_groups)]) /
                       (num_total - num_groups) * 1.0)

    numerator = (num_total - num_groups) * math.log(pooled_variance) - (np.sum([(num_samples[i] - 1) *
                                                                                math.log(std_groups[i] ** 2)
                                                                                for i in range(num_groups)]))

    denominator = 1.0 + (1.0 / (3.0 * (num_groups - 1))) * (np.sum([1.0 / (i - 1.0) for i in num_samples]) -
                                                            (1 / float(num_total - num_groups)))
    statistical_bartlett = numerator / denominator

    p_value, cv_value = stats.get_p_value_chi2(statistical_bartlett, num_groups - 1, alpha=alpha)

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"

    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    # print("\t p_value = ", p_value)

    return statistical_bartlett, p_value, cv_value, hypothesis


def t_test_paired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
        Perform a paired t-test. This statistical test is used to compare the means of two related groups of samples,
        typically before and after a specific treatment or intervention. The test assumes that the differences between
        pairs are normally distributed.

        :param dataset: A pandas DataFrame with exactly two columns, each representing a different group/sample.
                        These groups must be related or paired in some way (e.g., measurements before and after an
                        intervention).
        :param alpha: The significance level for the test, default is 0.05.
        :param verbose: A boolean (True or False). If True, prints the test statistic, rejected value, p-value, and
                        hypothesis.

        :return: A tuple containing the following:
            - statistical_t: The t-test statistic. A higher absolute value indicates a greater difference between the
              paired groups.
            - rejected_value: The critical value for the test at the specified alpha level.
            - p_value: The p-value for the hypothesis test (currently not calculated and set to None).
            - hypothesis: A string stating the conclusion of the test based on the test statistic and alpha.
                          It indicates whether the null hypothesis (no difference between means) can be rejected or not.

        Note: The paired t-test is appropriate for comparing two means from the same group or individual under two
        different conditions. The test statistic is calculated by dividing the mean difference between paired
        observations by the standard error of the mean difference. The test is sensitive to the normality assumption of
        the differences between pairs.
    """
    if dataset.shape[1] != 2:
        raise "Error: T Test need two samples"

    names_groups = list(dataset.columns)
    num_samples = dataset.shape[0]
    mean_samples = np.mean(dataset[names_groups[0]] - dataset[names_groups[1]])
    std_samples = np.std(dataset[names_groups[0]] - dataset[names_groups[1]])

    standard_error_of_the_mean = std_samples / math.sqrt(num_samples)

    statistical_t = mean_samples / standard_error_of_the_mean

    # TODO Calculate P-Valor T Distribution with alpha, degrees_of_freedom = num_samples - 1 (Revisar tras hablar)

    rejected_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_t < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)
    return statistical_t, rejected_value, p_value, hypothesis


def t_test_unpaired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
        Perform an unpaired (independent) t-test. This statistical test is used to compare the means of two unrelated
        groups of samples to determine if there is a statistically significant difference between the two means. It's
        appropriate when the two groups are independent of each other.

        :param dataset: A pandas DataFrame with exactly two columns, each representing a different group/sample.
                        These groups must be independent or unpaired.
        :param alpha: The significance level for the test, default is 0.05.
        :param verbose: A boolean (True or False). If True, prints the test statistic, rejected value, p-value, and
                        hypothesis.

        :return: A tuple containing the following:
            - statistical_t: The t-test statistic. A higher absolute value indicates a greater difference between the
                             group means.
            - rejected_value: The critical value for the test at the specified alpha level.
            - p_value: The p-value for the hypothesis test (currently not calculated and set to None).
            - hypothesis: A string stating the conclusion of the test based on the test statistic and alpha.
                          It indicates whether the null hypothesis (no difference between means) can be rejected or not.

        Note: The unpaired t-test assumes that the two groups have equal variances and that the samples are randomly
        drawn. The test statistic is calculated by taking the difference between the two group means and dividing by the
        standard error of the mean difference. The test is sensitive to the normality assumption of the sample data.
    """
    if dataset.shape[1] != 2:
        raise "Error: T Test need two samples"

    names_groups = list(dataset.columns)
    num_samples = dataset.shape[0]

    numerator = (num_samples - 1) * (np.std(dataset[names_groups[0]]) ** 2 + np.std(dataset[names_groups[1]]) ** 2)
    denominator = num_samples + num_samples - 2

    estimated_of_std = math.sqrt(numerator / denominator)

    std_of_the_mean_value = estimated_of_std * math.sqrt(1 / float(num_samples) + 1 / float(num_samples))

    statistical_t = np.mean(dataset[names_groups[0]]) - np.mean(dataset[names_groups[1]]) / std_of_the_mean_value

    # TODO Calculate P-Valor T Distribution with alpha, degrees_of_freedom = num_samples - 1 (Revisar tras hablar)
    rejected_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    p_value = None
    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_t < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)

    return statistical_t, rejected_value, p_value, hypothesis


def anova_cases(dataset: pd.DataFrame, alpha: float = 0.05):
    """
        Perform an ANOVA (Analysis of Variance) test. This statistical test is used to compare the means of two or more
        groups to determine if at least one group mean is significantly different from the others. It's commonly used
        when there are three or more groups.

        :param dataset: A pandas DataFrame where each column represents a different group/sample. The first column is
                        ignored.
        :param alpha: The significance level for the test, default is 0.05.

        :return: A tuple containing the following:
            - summary_results: A pandas DataFrame with a summary of each group's mean, standard deviation, and standard
                               error.
            - anova_results: A pandas DataFrame with the ANOVA results, including degrees of freedom, sum of squares,
                             mean square, F-statistic, and rejected value for between groups and within groups.
            - statistical_f_anova: The F-statistic for the ANOVA test. A higher value suggests a greater difference
                                   between group means.
            - p_value: The p-value for the hypothesis test (currently not calculated and set to None).
            - rejected_value: The critical value for the test at the specified alpha level.
            - hypothesis: A string stating the conclusion of the test based on the F-statistic and alpha.

        Note: ANOVA tests the null hypothesis that all group means are equal. The test assumes that the groups are
        sampled from populations with normal distributions and equal variances. The F-statistic is calculated based on
        the ratio of variance between the groups to the variance within the groups. A significant F-statistic (p-value
        less than alpha) indicates that at least one group mean is significantly different.
    """
    if dataset.shape[1] < 2:
        raise "Error: Anova Test need at least two samples"

    names_groups = list(dataset.columns)
    names_groups = names_groups[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)
    sum_groups = []
    sum_square_groups = []
    for i in names_groups:
        sum_groups.append(np.sum(dataset[i]))
        sum_square_groups.append(np.sum(dataset[i] ** 2))

    sum_x_t = sum(sum_groups)

    ss_bg = np.sum([(sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)]) - ((sum_x_t ** 2) / num_total)
    df_bg = num_groups - 1

    ss_wg = np.sum([sum_square_groups[i] - (sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)])
    df_wg = num_total - num_groups

    ms_bg = ss_bg / df_bg
    ms_wg = ss_wg / df_wg

    statistical_f_anova = ms_bg / ms_wg

    # TODO Calculate P-Valor F Dist with alpha df_numerator (df_bg) df_denominator (df_wg) (Revisar tras hablar)
    # p_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    rejected_value = stats.get_cv_f_distribution(df_bg, df_wg, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    summary_results = [(np.mean(dataset[i].to_numpy()), np.std(dataset[i].to_numpy()),
                        np.std(dataset[i].to_numpy()) / math.sqrt(int(dataset[i].shape[0]))) for i in names_groups]

    mean_groups, std_dev_groups, std_error_groups = zip(*summary_results)
    content = {"Groups": names_groups, "Nº Samples": num_samples, "Mean": mean_groups, "Std. Dev.": std_dev_groups,
               "Std. Error": std_error_groups}
    summary_results = pd.DataFrame(content)
    sources = ["Between Groups", "Within Groups", "Total"]
    df_list = [df_bg, df_wg, num_total - 1]
    sum_squares = [ss_bg, ss_wg, ss_bg + ss_wg]
    mean_square = [ms_bg, ms_wg, ms_bg + ms_wg]
    f_stats = [statistical_f_anova, "", ""]
    rejected_values = [rejected_value, "", ""]
    anova_results = {"Source": sources, "Degrees of Freedom (DF)": df_list, "Sum of Squares (SS)": sum_squares,
                     "Mean Square (MS)": mean_square, "F-Stat": f_stats, "Rejected Value": rejected_values}
    anova_results = pd.DataFrame(anova_results)
    
    return [summary_results, anova_results], statistical_f_anova, p_value, rejected_value, hypothesis


def anova_within_cases(dataset: pd.DataFrame, alpha: float = 0.05):
    """
        Perform a within-subject ANOVA (Analysis of Variance). This type of ANOVA is used when the same subjects are
        used for each treatment (i.e., the subjects are subjected to repeated measures). This test is beneficial for
        analyzing the effects of different conditions or treatments on a single group of subjects.

        :param dataset: A pandas DataFrame where each column represents a different condition/treatment for the same
                        subjects. The first column is typically used for subject identification and is ignored in the
                        analysis.
        :param alpha: The significance level for the test, default is 0.05.

        :return: A tuple containing the following:
            - summary_results: A pandas DataFrame with a summary of each group's mean, standard deviation, and standard
                               error.
            - anova_results: A pandas DataFrame with the ANOVA results, including degrees of freedom, sum of squares,
                             mean square, F-statistic, and rejected value for different sources of variance (between
                             conditions, between subjects, etc.).
            - statistical_f_anova: The F-statistic for the ANOVA test. A higher value suggests a significant difference
                                   between conditions or treatments.
            - p_value: The p-value for the hypothesis test (currently not calculated and set to None).
            - rejected_value: The critical value for the test at the specified alpha level.
            - hypothesis: A string stating the conclusion of the test based on the F-statistic and alpha.

        Note: Within-subject ANOVA controls for potential variability among subjects, as each subject serves as their
        own control. This test separates the variance due to the interaction between subjects and conditions from the
        variance due to differences between conditions and residual variance. It assumes sphericity, which implies that
        the variances of the differences between all possible pairs of conditions are equal.
    """
    if dataset.shape[1] < 2:
        raise "Error: Anova Test need at least two samples"

    names_groups = list(dataset.columns)
    names_groups = names_groups[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)
    sum_groups = []
    sum_square_groups = []
    for i in names_groups:
        sum_groups.append(np.sum(dataset[i]))
        sum_square_groups.append(np.sum(dataset[i] ** 2))
    sum_s_i = dataset[names_groups].sum(axis=1)
    sum_x_t = sum(sum_groups)
    sum_x_square_t = sum(sum_square_groups)

    ss_bc = np.sum([(sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)]) - ((sum_x_t ** 2) / num_total)
    df_bc = num_groups - 1
    ms_bc = ss_bc / df_bc

    ss_bs = np.sum([i ** 2 / num_groups for i in sum_s_i]) - ((sum_x_t ** 2) / num_total)
    df_bs = num_samples[0] - 1
    ms_bs = ss_bs / df_bs

    ss_res = sum_x_square_t - ss_bc - ss_bs
    df_res = (num_samples[0] - 1) * (num_groups - 1)
    ms_res = ss_res / df_res

    statistical_f_anova = ms_bc / ms_res

    # TODO Calculate P-Valor F Dist with alpha and df_numerator (df_bc) df_denominator (df_res)
    # p_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    rejected_value = stats.get_cv_f_distribution(df_bc, df_res, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    summary_results = [(np.mean(dataset[i].to_numpy()), np.std(dataset[i].to_numpy()),
                        np.std(dataset[i].to_numpy()) / math.sqrt(int(dataset[i].shape[0]))) for i in names_groups]

    mean_groups, std_dev_groups, std_error_groups = zip(*summary_results)
    content = {"Groups": names_groups, "Nº Samples": num_samples, "Mean": mean_groups, "Std. Dev.": std_dev_groups,
               "Std. Error": std_error_groups}
    summary_results = pd.DataFrame(content)
    
    sources = ["Between Conditions", "Between Subjects", "Residual", "Total"]
    df_list = [df_bc, df_bs, df_res, df_bc + df_bs + df_res]
    sum_squares = [ss_bc, ss_bs, ss_res, ss_bc + ss_bs + ss_res]
    mean_square = [ms_bc, ms_bs, ms_res, ms_bc + ms_bs + ms_res]
    f_stats = [statistical_f_anova, "", "", ""]
    rejected_values = [rejected_value, "", "", ""]
    anova_results = {"Source": sources, "Degrees of Freedom (DF)": df_list, "Sum of Squares (SS)": sum_squares,
                     "Mean Square (MS)": mean_square, "F-Stat": f_stats, "Rejected Value": rejected_values}
    anova_results = pd.DataFrame(anova_results)
     
    return [summary_results, anova_results], statistical_f_anova, p_value, rejected_value, hypothesis

# PRUEBAS UNITARIAS DE LAS FUNCIONES QUITAR ESTO DE AQUI MÁS ADELANTE


def anova_test(dataset: pd.DataFrame, alpha: float = 0.05):
    if dataset.shape[1] < 2:
        raise "Error: Anova Test need at least two samples"

    names_groups = list(dataset.columns)
    names_groups = names_groups[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)
    sum_groups = []
    sum_square_groups = []
    for i in names_groups:
        sum_groups.append(np.sum(dataset[i]))
        sum_square_groups.append(np.sum(dataset[i] ** 2))

    sum_x_t = sum(sum_groups)

    ss_bg = np.sum([(sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)]) - ((sum_x_t ** 2) / num_total)
    df_bg = num_groups - 1

    ss_wg = np.sum([sum_square_groups[i] - (sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)])
    df_wg = num_total - num_groups

    ms_bg = ss_bg / df_bg
    ms_wg = ss_wg / df_wg

    statistical_f_anova = ms_bg / ms_wg

    # TODO Calculate P-Valor F Dist with alpha df_numerator (df_bg) df_denominator (df_wg) (Revisar tras hablar)
    # p_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    rejected_value = stats.get_cv_f_distribution(df_bg, df_wg, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return statistical_f_anova, p_value, rejected_value, hypothesis


def anova_within_cases_test(dataset: pd.DataFrame, alpha: float = 0.05):
    if dataset.shape[1] < 2:
        raise "Error: Anova Test need at least two samples"

    names_groups = list(dataset.columns)
    names_groups = names_groups[1:]
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)
    sum_groups = []
    sum_square_groups = []
    for i in names_groups:
        sum_groups.append(np.sum(dataset[i]))
        sum_square_groups.append(np.sum(dataset[i] ** 2))
    sum_s_i = dataset.sum(axis=1)
    sum_x_t = sum(sum_groups)
    sum_x_square_t = sum(sum_square_groups)

    ss_bc = np.sum([(sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)]) - ((sum_x_t ** 2) / num_total)
    df_bc = num_groups - 1
    ms_bc = ss_bc / df_bc

    # ss_bs = np.sum([i ** 2 / num_groups for i in sum_s_i]) - ((sum_x_t ** 2) / num_total)
    # df_bs = num_samples[0] - 1
    # ms_bs = ss_bs / df_bs

    ss_res = sum_x_square_t - np.sum([(sum_groups[i] ** 2) / num_samples[i] for i in range(num_groups)]) - np.sum(
        [i ** 2 / num_groups for i in sum_s_i]) + ((sum_x_t ** 2) / num_total)
    df_res = (num_samples[0] - 1) * (num_groups - 1)
    ms_res = ss_res / df_res

    statistical_f_anova = ms_bc / ms_res

    # TODO Calculate P-Valor F Dist with alpha and df_numerator (df_bc) df_denominator (df_res)
    # p_value = stats.get_cv_f_distribution(num_groups, num_samples[0] - num_groups, alpha=alpha)
    rejected_value = stats.get_cv_f_distribution(df_bc, df_res, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return statistical_f_anova, rejected_value, p_value, hypothesis


def test_bartlett():
    data = pd.read_csv("prueba.csv")
    levene_test(data)


def test_levene():
    data = pd.read_csv("prueba.csv")
    levene_test(data)


def test_d_agostino_pearson():
    data = [43, 36, 43, 41, 37, 37, 43, 40]
    d_agostino_pearson(data)


def test_shapiro_wilk():
    data = [43, 36, 43, 41, 37, 37, 43, 40]
    shapiro_wilk_normality(data)
