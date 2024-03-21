import pandas as pd
import numpy as np
import math

from . import stats


def levene_test(dataset: pd.DataFrame, alpha: float = 0.05, center: str = 'mean'):
    """
    Perform the Levene test for equality of variances. This test is used to assess whether 
    the variances of two or more groups are equal. It is an essential test before conducting 
    ANOVA, as ANOVA assumes homogeneity of variances.

    Parameters
    ----------
    dataset : pandas DataFrame
        A DataFrame where each column represents a different group/sample. 
        The first column is ignored.
    alpha : float, optional
        The significance level for the test, default is 0.05.
    center : {'mean', 'median', 'trimmed'}, optional
        The method for calculating the center of each group. Default is 'mean'.

    Returns
    -------
    statistic_levene : float
        The Levene test statistic. A higher value indicates a greater likelihood 
        of differing variances.
    p_value : float
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) 
        rejects the null hypothesis, suggesting that the variances across groups 
        are not equal.
    rejected_value : float
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic 
        and alpha. It indicates whether the null hypothesis of equal variances 
        can be rejected or not.

    Notes
    -----
    The Levene test is robust to non-normal distributions, making it preferable to Bartlett's 
    test when data are not normally distributed. The choice of 'center' parameter (mean, median, 
    trimmed) can affect the test's sensitivity to departures from normality. The test statistic 
    is computed based on the absolute deviations from the group centers and then compared against 
    an F-distribution to obtain the p-value.
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

    rejected_value = stats.get_cv_f_distribution(num_groups - 1, num_samples[0] - num_groups, alpha=alpha)
    p_value = stats.get_p_value_f(statistic_levene, num_groups - 1, num_total - num_groups)

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if p_value > alpha:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha}(Same distributions)"

    return statistic_levene, p_value, rejected_value, hypothesis


def bartlett_test(dataset: pd.DataFrame, alpha: float = 0.05):
    """
    Perform Bartlett's test for homogeneity of variances. This test checks the null 
    hypothesis that all input samples (represented as columns in the dataset) come 
    from populations with equal variances. It is commonly used before conducting ANOVA, 
    as equal variances are an assumption of ANOVA.

    Parameters
    ----------
    dataset : pandas DataFrame
        A DataFrame where each column represents a different group/sample. 
        The first column is ignored.
    alpha : float, optional
        The significance level for the test, default is 0.05.

    Returns
    -------
    statistical_bartlett : float
        Bartlett's test statistic. A higher value indicates a greater likelihood 
        of differing variances.
    p_value : float
        The p-value for the hypothesis test. A small p-value (typically ≤ 0.05) 
        rejects the null hypothesis, suggesting that the variances across groups 
        are not equal.
    cv_value : float
        Critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic 
        and alpha. It indicates whether the null hypothesis of equal variances 
        can be rejected or not.

    Notes
    -----
    Bartlett's test is sensitive to departures from normality. Therefore, if the data 
    are not normally distributed, Levene's test is a more appropriate choice. The test 
    statistic is compared against a chi-squared distribution to obtain the p-value. 
    The formula for the test statistic takes into account the number of groups and 
    the total number of samples.
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

    hypothesis = f"Reject H0 with alpha = {alpha} (Different Variances)"
    if p_value > alpha:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same Variances)"

    return statistical_bartlett, p_value, cv_value, hypothesis
