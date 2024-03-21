import numpy as np
import math
import pandas as pd

from . import stats


def t_test_paired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform a paired t-test. This statistical test is used to compare the means of two related groups of samples,
    typically before and after a specific treatment or intervention. The test assumes that the differences between
    pairs are normally distributed.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different group/sample. These groups must be related 
        or paired in some way (e.g., measurements before and after an intervention).
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        A boolean indicating whether to print detailed results. If True, prints the test statistic, rejected value, 
        p-value, and hypothesis. Default is False.

    Returns
    -------
    statistical_t : float
        The t-test statistic. A higher absolute value indicates a greater difference between the paired groups.
    p_value : float
        The p-value for the hypothesis test (currently not calculated and set to None).
    rejected_value : float
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic and alpha. It indicates whether the null 
        hypothesis (no difference between means) can be rejected or not.

    Note
    ----
    The paired t-test is appropriate for comparing two means from the same group or individual under two different
    conditions. The test statistic is calculated by dividing the mean difference between paired observations by the
    standard error of the mean difference. The test is sensitive to the normality assumption of the differences between
    pairs.
    """

    if dataset.shape[1] != 2:
        raise "Error: T Test need two samples"

    names_groups = list(dataset.columns)
    num_samples = dataset.shape[0]
    diff = dataset[names_groups[0]] - dataset[names_groups[1]]

    sum_d = diff.sum()

    sum_d_2 = (diff ** 2).sum()
    mean_d = sum_d / num_samples
    s_d = math.sqrt((sum_d_2 - (sum_d ** 2) / num_samples) / (num_samples - 1))
    s_d = s_d / math.sqrt(num_samples)
    statistical_t = mean_d / s_d
    rejected_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    p_value = 2 * stats.get_p_value_t(statistical_t, num_samples - 1)
    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if statistical_t < rejected_value:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)

    return statistical_t, p_value, rejected_value, hypothesis


def t_test_unpaired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform an unpaired (independent) t-test. This statistical test is used to compare the means of two unrelated
    groups of samples to determine if there is a statistically significant difference between the two means. It's
    appropriate when the two groups are independent of each other.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different group/sample. These groups must be 
        independent or unpaired.
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        A boolean indicating whether to print detailed results. If True, prints the test statistic, rejected value, 
        p-value, and hypothesis. Default is False.

    Returns
    -------
    statistical_t : float
        The t-test statistic. A higher absolute value indicates a greater difference between the group means.
    p_value : float
        The p-value for the hypothesis test (currently not calculated and set to None).
    rejected_value : float
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic and alpha. It indicates whether the null 
        hypothesis (no difference between means) can be rejected or not.

    Note
    ----
    The unpaired t-test assumes that the two groups have equal variances and that the samples are randomly drawn. 
    The test statistic is calculated by taking the difference between the two group means and dividing by the standard 
    error of the mean difference. The test is sensitive to the normality assumption of the sample data.
    """

    if dataset.shape[1] != 2:
        raise "Error: T Test need two samples"

    names_groups = list(dataset.columns)
    num_samples = dataset.shape[0]
    sum_1 = dataset[names_groups[0]].sum()
    sum_2 = dataset[names_groups[1]].sum()

    square_sum_1 = (dataset[names_groups[0]] ** 2).sum()
    square_sum_2 = (dataset[names_groups[1]] ** 2).sum()

    x_1_mean = sum_1 / num_samples
    x_2_mean = sum_2 / num_samples

    s_1 = (square_sum_1 - (sum_1 ** 2 / num_samples)) / (num_samples - 1)
    s_2 = (square_sum_2 - (sum_2 ** 2 / num_samples)) / (num_samples - 1)

    statistical_t = (x_1_mean - x_2_mean) / math.sqrt(s_1 / num_samples + s_2 / num_samples)

    rejected_value = stats.get_cv_t_distribution(num_samples * 2 - 2, alpha=alpha)
    p_value = 2 * stats.get_p_value_t(statistical_t, num_samples * 2 - 2)
    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if statistical_t < rejected_value:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)

    return statistical_t, p_value, rejected_value, hypothesis


def anova_cases(dataset: pd.DataFrame, alpha: float = 0.05):
    """
    Perform an ANOVA (Analysis of Variance) test. This statistical test is used to compare the means of two or more
    groups to determine if at least one group mean is significantly different from the others. It's commonly used
    when there are three or more groups.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame where each column represents a different group/sample. The first column is ignored.
    alpha : float, optional
        The significance level for the test. Default is 0.05.

    Returns
    -------
    summary_results : pandas.DataFrame
        A DataFrame with a summary of each group's mean, standard deviation, and standard error.
    anova_results : pandas.DataFrame
        A DataFrame with the ANOVA results, including degrees of freedom, sum of squares, mean square, F-statistic, 
        and rejected value for between groups and within groups.
    statistical_f_anova : float
        The F-statistic for the ANOVA test. A higher value suggests a greater difference between group means.
    p_value : float
        The p-value for the hypothesis test (currently not calculated and set to None).
    rejected_value : float
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the F-statistic and alpha.

    Note
    ----
    ANOVA tests the null hypothesis that all group means are equal. The test assumes that the groups are
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

    rejected_value = stats.get_cv_f_distribution(df_bg, df_wg, alpha=alpha)
    p_value = stats.get_p_value_f(statistical_f_anova, df_bg, df_wg)

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

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
    
    return summary_results, anova_results, statistical_f_anova, p_value, rejected_value, hypothesis


def anova_within_cases(dataset: pd.DataFrame, alpha: float = 0.05):
    """
    Perform a within-subject ANOVA (Analysis of Variance). This type of ANOVA is used when the same subjects are
    used for each treatment (i.e., the subjects are subjected to repeated measures). This test is beneficial for
    analyzing the effects of different conditions or treatments on a single group of subjects.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame where each column represents a different condition/treatment for the same subjects. The first 
        column is typically used for subject identification and is ignored in the analysis.
    alpha : float, optional
        The significance level for the test. Default is 0.05.

    Returns
    -------
    summary_results : pandas.DataFrame
        A DataFrame with a summary of each group's mean, standard deviation, and standard error.
    anova_results : pandas.DataFrame
        A DataFrame with the ANOVA results, including degrees of freedom, sum of squares, mean square, F-statistic, 
        and rejected value for different sources of variance (between conditions, between subjects, etc.).
    statistical_f_anova : float
        The F-statistic for the ANOVA test. A higher value suggests a significant differensce between conditions or
        treatments.
    p_value : float
        The p-value for the hypothesis test (currently not calculated and set to None).
    rejected_value : float
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the F-statistic and alpha.

    Note
    ----
    Within-subject ANOVA controls for potential variability among subjects, as each subject serves as their own control. 
    This test separates the variance due to the interaction between subjects and conditions from the variance due to 
    differences between conditions and residual variance. It assumes sphericity, which implies that the variances of the 
    differences between all possible pairs of conditions are equal.
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

    ss_t = sum_x_square_t - (sum_x_t ** 2) / num_total
    ss_res = ss_t - ss_bc - ss_bs
    df_res = (num_samples[0] - 1) * (num_groups - 1)
    ms_res = ss_res / df_res

    statistical_f_anova = ms_bc / ms_res
    rejected_value = stats.get_cv_f_distribution(df_bc, df_res, alpha=alpha)
    p_value = stats.get_p_value_f(statistical_f_anova, df_bc, df_res)

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if statistical_f_anova < rejected_value:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

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
     
    return summary_results, anova_results, statistical_f_anova, p_value, rejected_value, hypothesis
