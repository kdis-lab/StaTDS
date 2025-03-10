import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

from . import stats


class LibraryError(Exception):
    def __init__(self, message):
        super().__init__(message)


# -------------------- Test Two Groups -------------------- #
def wilcoxon(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform the Wilcoxon signed-rank test. This non-parametric test is used to compare two related samples, matched
    samples, or repeated measurements on a single sample to assess whether their population mean ranks differ. It is
    an alternative to the paired Student's t-test when the data is not normally distributed.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different condition or time point for the same
        subjects.
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        If True, prints the detailed results table.

    Returns
    -------
    w_wilcoxon : float
        The Wilcoxon test statistic, which is the smallest of the sums of the positive and negative ranks.
    p_value : float or None
        The p-value for the hypothesis test (only for large sample sizes, otherwise None).
    cv_alpha_selected : float or None
        The critical value for the test at the specified alpha level (only for small sample sizes, otherwise None).
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, or p-value and alpha.

    Note
    ----
    The Wilcoxon signed-rank test makes fewer assumptions than the t-test and is appropriate when the data
    are not normally distributed. It ranks the absolute differences between pairs, then compares these ranks.
    The test is sensitive to ties and has different procedures for small and large sample sizes. For large samples,
    the test statistic is approximately normally distributed, allowing the use of normal approximation for p-value
    calculation.
    """
    if dataset.shape[1] != 2:
        raise "Error: The test only needs two samples"

    results_table = dataset.copy()
    columns = list(dataset.columns)
    differences_results = dataset[columns[0]] - dataset[columns[1]]
    absolute_dif = differences_results.abs()
    absolute_dif = absolute_dif.sort_values()
    results_wilconxon = {"index": [], "dif": [], "rank": [], "R": []}
    rank = 0.0
    tied_ranges = not (len(set(absolute_dif)) == absolute_dif.shape[0])
    for index in absolute_dif.index:
        if math.fabs(0 - absolute_dif[index]) < 1e-10:
            continue
        rank += 1.0
        results_wilconxon["index"] += [index]
        results_wilconxon["dif"] += [differences_results[index]]
        results_wilconxon["rank"] += [rank]
        results_wilconxon["R"] += ["+" if differences_results[index] > 0 else "-"]

    df = pd.DataFrame(results_wilconxon)
    df = df.set_index("index")
    df = df.sort_index()
    results_table = pd.concat([results_table, df], axis=1)

    tie_sum = 0

    if tied_ranges:
        vector = [abs(i) for i in results_table["dif"]]

        counts = {}
        for number in vector:
            try:
                counts[number] = counts[number] + 1
            except KeyError:
                counts[number] = 1

        ranks = results_table["rank"].to_numpy()
        for index, number in enumerate(vector):
            if counts[number] > 1:
                rank_sum = sum(ranks[i] for i, x in enumerate(vector) if x == number)
                average_rank = rank_sum / counts[number]
                for i, x in enumerate(vector):
                    if x == number:
                        ranks[i] = average_rank
        tie_sizes = np.array(list(counts.values()))
        tie_sum = (tie_sizes ** 3 - tie_sizes).sum()

    if verbose:
        print(results_table)

    r_plus = results_table[results_table.R == "+"]["rank"].sum()
    r_minus = results_table[results_table.R == "-"]["rank"].sum()

    w_wilcoxon = min([r_plus, r_minus])
    num_problems = results_table.shape[0] - (results_table.R.isna().sum())
    mean_wilcoxon = (num_problems * (num_problems + 1)) / 4.0

    std_wilcoxon = num_problems * (num_problems + 1) * ((2 * num_problems) + 1)
    std_wilcoxon = math.sqrt(std_wilcoxon / 24.0 - (tie_sum / 48))
    z_wilcoxon = (w_wilcoxon - mean_wilcoxon) / std_wilcoxon

    cv_alpha_selected = stats.get_cv_willcoxon(num_problems, alpha)

    p_value = 2 * stats.get_p_value_normal(z_wilcoxon)

    # if num_problems > 25:
    hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"

    return w_wilcoxon, p_value, cv_alpha_selected, hypothesis


def binomial(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform the Binomial Sign Test. This non-parametric test is used to determine if there is a significant difference
    between the medians of two dependent samples. It serves as an alternative to the paired t-test and Wilcoxon
    signed-rank test, particularly useful when the data does not meet the assumptions of these tests or when the data
    is on an ordinal scale.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different condition or time point for the same
        subjects.
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        If True, prints the detailed results table.

    Returns
    -------
    statistical_binomial : float
        The Binomial test statistic, which is the largest of the counts of positive or negative differences.
    p_value : float
        The p-value for the hypothesis test, calculated from the binomial distribution.
    cv_value : float or None
        The critical value for the test at the specified alpha level (not calculated in this function, hence None).
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic and p-value in comparison to alpha.

    Note
    ----
    The Binomial Sign Test counts the number of positive and negative differences between paired observations and then
    tests if the observed proportion of positive (or negative) differences is significantly different from 0.5
    (no difference). The test assumes that the distribution of differences is symmetric around the median and ignores
    pairs with no difference.
    """
    if dataset.shape[1] != 2:
        print("Error: The test only needs two samples")

    columns = list(dataset.columns)
    df = dataset.copy()
    df["Difference"] = df[columns[0]] - df[columns[1]]
    df['Sign'] = df['Difference'].apply(lambda x: '+' if x > 0 else '-' if x < 0 else '0')

    counts = dict(df['Sign'].value_counts())
    if verbose:
        print(df)

    try:
        diffence_plus = counts["+"]
    except KeyError:
        diffence_plus = 0

    try:
        diffence_minus = counts["-"]
    except KeyError:
        diffence_minus = 0

    statistical_binomial = max(diffence_plus, diffence_minus)
    num_samples = diffence_plus + diffence_minus
    # Calculate the p-value of binomial distribution
    p_value = stats.get_p_value_binomial(num_samples, statistical_binomial)

    hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    cv_value = None

    return statistical_binomial, p_value, cv_value, hypothesis


def mannwhitneyu(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    """
    Perform the Mann-Whitney U Test, also known as the Wilcoxon rank-sum test. This non-parametric test is utilized to
    determine whether there is a significant difference between the distributions of two independent samples. It serves
    as an alternative to the independent t-test, particularly when the data does not satisfy the assumptions of the
    t-test.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with exactly two columns, each representing a different independent sample.
    alpha : float, optional
        The significance level for the test. Default is 0.05.
    verbose : bool, optional
        If True, prints the detailed results table.

    Returns
    -------
    z_value : float
        The z-value computed from the U statistic.
    cv_alpha_selected : float or None
        The critical value for the test at the specified alpha level (not calculated in this function, hence None).
    p_value : float
        The p-value for the hypothesis test, calculated from the normal approximation of the U distribution.
    hypothesis : str
        A string stating the conclusion of the test based on the z-value and p-value in comparison to alpha.

    Note
    ----
    The Mann-Whitney U test ranks all observations from both groups together and then compares the sum of ranks in each
    group. It is suitable for ordinal data and is robust against non-normal distributions. The test assumes that the two
    groups are independent and that the observations are ordinal or continuous. The normal approximation for the p-value
    calculation is valid for large sample sizes.
    """
    if dataset.shape[1] != 2:
        raise Exception("Error: The test only needs two samples")

    rankings = dataset.stack().rank(method='average').unstack()
    columns = list(dataset.columns)
    n1 = n2 = dataset.shape[0]

    rank_1 = rankings.sum()[columns[0]]
    rank_2 = rankings.sum()[columns[1]]

    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - rank_1

    u2 = n1 * n2 + (n2 * (n2 + 1)) / 2 - rank_2

    if n1 * n2 != u1 + u2:
        raise Exception("Error to calculate U1 y U2")

    statistical_u = min(u1, u2)

    if verbose:
        print(rankings)

    numerator = statistical_u - ((n1 * n2) / 2)
    n = n1 + n2

    # Tie Correction for the normal approximation

    rank_values = list(rankings.stack())
    sorted_values = np.sort(rank_values)
    value_changes_idx = np.nonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])[0]
    tie_sizes = np.diff(value_changes_idx).astype(np.float64)
    tie_sum = (tie_sizes ** 3 - tie_sizes).sum()

    denominator = math.sqrt(n1*n2/12 * ((n + 1) - tie_sum/(n*(n-1))))

    numerator = abs(numerator) - 0.5

    z_value = numerator / denominator

    p_value = 2 * stats.get_p_value_normal(z_value)

    hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    cv_value = None

    return u2, p_value, cv_value, hypothesis

# -------------------- Test Two Groups -------------------- #


# -------------------- Test Multiple Groups -------------------- #
def friedman(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False,
             apply_correction: bool = False) -> object:
    """
    Perform the Friedman test, a non-parametric statistical test similar to the parametric ANOVA, but for
    repeated measures. The Friedman test is used to detect differences in treatments across multiple test
    attempts. It ranks the treatments for each block (or subject), then considers these ranks.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with the first column as the block or subject identifier and the remaining
        columns as different treatments or conditions.
    alpha : float, optional
        The significance level for the test, default is 0.05.
    minimize : bool, optional
        Determines whether the metric of interest should be minimized or maximized. If True, the metric is to be
        minimized; if False, it is to be maximized. This parameter is crucial for tailoring the function's behavior to
        the specific nature of the data or the goal of the analysis.
    verbose : bool, optional
        If True, prints the detailed results table including ranks.
    apply_correction : bool, optional
        If True, apply Tie correction for the Friedman two-way analysis of variance by rank

    Returns
    -------
    rankings_with_label : dict
        A dictionary with the average ranks of each treatment or condition.
    statistic_friedman : float
        The Friedman test statistic, which is chi-squared distributed under the null
        hypothesis.
    p_value : float
        The p-value for the hypothesis test.
    cv_alpha_selected : float or None
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, and
        alpha.

    Note
    ----
    The Friedman test is appropriate when data is ordinal and the assumptions of parametric tests (like ANOVA)
    are not met. It considers the ranks of the treatments within each block, summing these ranks, and then analyzing
    the sums' distribution. The test is robust against non-normal distributions and is ideal for small sample sizes.
    """
    columns_names = list(dataset.columns)
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    ranks = []

    for i in dataset.index:
        sr = dataset.loc[i][columns_names[1:]]
        ranks.append(sr.rank(method='average', ascending=minimize).tolist())

    rank_values = ranks
    df = pd.DataFrame(ranks, columns=columns_names[1:])
    ranks = [df[i].sum() for i in df.columns]
    row_ranks = {i: [j / num_cases] for i, j in zip(df.columns, ranks)}
    row_ranks = pd.DataFrame(row_ranks)
    df = pd.concat([df, row_ranks])
    df.reset_index(inplace=True, drop=True)
    new_index = [i for i in dataset[columns_names[0]].values]
    new_index += ["R_mean"]
    new_index = pd.DataFrame({columns_names[0]: new_index})
    df = pd.concat([df, new_index], axis=1)
    df.set_index(columns_names[0], inplace=True)
    ranks_sum = sum([i ** 2 for i in ranks])
    stadistic_friedman = (((12 / (num_cases * num_algorithm * (num_algorithm + 1))) * ranks_sum) - 3 * num_cases *
                          (num_algorithm + 1))
    if apply_correction:
        # [1] Handbook-of-parametric-and-nonparametric-statistical-procedures
        # Tie correction for the Friedman two-way analysis of variance by rank:
        total_ties = []
        for rank in rank_values:
            sorted_values = np.sort(rank)
            value_changes_idx = np.nonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])[0]
            tie_sizes = np.diff(value_changes_idx).astype(np.float64)
            total_ties += list(tie_sizes)

        tie_sizes = np.array(total_ties)

        correction = 1.0 - (tie_sizes ** 3 - tie_sizes).sum() / (num_cases * (num_algorithm ** 3 - num_algorithm))

        stadistic_friedman = stadistic_friedman / correction

    hypothesis_state = True
    if verbose:
        print(df)
    rankings_with_label = {j: i / num_cases for i, j in zip(ranks, columns_names[1:])}

    if num_cases > 15 or num_algorithm >= 3:  # > 4
        # P-value = P(chi^2_{k-1} >= Q)
        p_value, critical_value = stats.get_p_value_chi2(stadistic_friedman, num_algorithm-1, alpha=alpha)
        if p_value < alpha:
            hypothesis_state = False
    else:
        p_value, critical_value = None, stats.get_cv_q_distribution(num_cases, num_algorithm-1, alpha)
        if stadistic_friedman <= critical_value:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = False

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if hypothesis_state:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

    return rankings_with_label, stadistic_friedman, p_value, critical_value, hypothesis


def iman_davenport(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False):
    """
    Perform the Iman-Davenport test, a non-parametric statistical test that is a modification of the Friedman test.
    This test is used when comparing more than two treatments or conditions across multiple blocks or subjects. 
    Unlike the Friedman test, the Iman-Davenport test converts the Friedman statistic into an F-distribution 
    providing a better approximation in certain cases, especially when dealing with small sample sizes.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with the first column as the block or subject identifier and the remaining
        columns as different treatments or conditions.
    alpha : float, optional
        The significance level for the test, default is 0.05.
    minimize : bool, optional
        Determines whether the metric of interest should be minimized or maximized. If True, the metric is to be
        minimized; if False, it is to be maximized. This parameter is crucial for tailoring the function's behavior to
        the specific nature of the data or the goal of the analysis.
    verbose : bool, optional
        If True, prints the detailed results table including ranks.

    Returns
    -------
    rankings_with_label : dict
        A dictionary with the average ranks of each treatment or condition.
    statistic_iman_davenport : float
        The Iman-Davenport test statistic, which follows an F-distribution under the null
        hypothesis.
    reject_value : tuple
        A tuple containing the critical value for the test and the p-value (if applicable).
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, and
        alpha.

    Note
    ----
    The Iman-Davenport test is particularly useful in cases where the assumptions of the Friedman test are 
    not fully met or when a more sensitive analysis is required. It is especially effective in small sample sizes 
    and is robust against non-normal distributions, similar to the Friedman test. However, it provides a more 
    accurate approximation to the F-distribution, making it preferable in certain statistical analyses.
    """
    rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha=alpha, minimize=minimize,
                                                                        verbose=verbose)

    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1

    f_f = ((num_cases - 1) * statistic) / (num_cases * (num_algorithm - 1) - statistic)

    p_value = stats.get_p_value_f(f_f, num_algorithm-1, (num_algorithm - 1)*(num_cases-1))

    hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"

    critical_value = None
    return rankings, f_f, p_value, critical_value, hypothesis


def friedman_aligned_ranks(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False):
    """
    Perform the Friedman Aligned Ranks test, an extension of the Friedman test. This test is used when dealing with
    multiple treatments or conditions over different subjects or blocks, especially in cases where the assumptions
    of the classical Friedman test may not hold. The test aligns the data by subtracting the mean across treatments
    for each subject before ranking.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with the first column as the block or subject identifier and the remaining
        columns as different treatments or conditions.
    alpha : float, optional
        The significance level for the test, default is 0.05.
    minimize : bool, optional
        Determines whether the metric of interest should be minimized or maximized. If True, the metric is to be
        minimized; if False, it is to be maximized. This parameter is crucial for tailoring the function's behavior to
        the specific nature of the data or the goal of the analysis.
    verbose : bool, optional
        If True, prints the detailed results table including aligned ranks.

    Returns
    -------
    rankings_with_label : dict
        A dictionary with the average ranks of each treatment or condition.
    statistic_friedman : float
        The Friedman Aligned Ranks test statistic.
    p_value : float
        The p-value for the hypothesis test.
    cv_alpha_selected : float or None
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, and
        alpha.

    Note
    ----
    The Friedman Aligned Ranks test modifies the standard Friedman test by aligning the data for each subject
    before ranking. This alignment is achieved by subtracting the average rank across treatments for each subject,
    making the test more robust to certain types of data irregularities, like outliers. It is appropriate for
    ordinal data and assumes that the groups are independent and identically distributed within each block.
    """
    columns_names = list(dataset.columns)
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    df = pd.DataFrame(columns=dataset.columns)

    df[columns_names[0]] = dataset[columns_names[0]]
    columns_names.pop(0)
    means_results = dataset[columns_names].mean(axis=1)

    for i in columns_names:
        df[i] = dataset[i] - means_results

    aligned_observations = sorted(df[columns_names].values.flatten().tolist())
    aligned_observations = list(reversed(aligned_observations)) if not minimize else aligned_observations
    df_aligned = df.copy()
    for index in df.index:
        for algorith in columns_names:
            v = df.loc[index][algorith]
            df.at[index, algorith] = aligned_observations.index(v) + 1 + (aligned_observations.count(v) - 1) / 2.
            # print(v, df.at[index, algorith])

    ranks_j = [(df[i].sum()) ** 2 for i in columns_names]
    ranks_i = df[columns_names].sum(axis=1)
    ranks_i = (ranks_i ** 2).sum()
    sum_ranks_j = sum(ranks_j)
    numerator = ((num_algorithm - 1) * (sum_ranks_j - ((num_algorithm * (num_cases ** 2) / 4.0) *
                                                       (num_algorithm * num_cases + 1) ** 2)))

    denominator = (num_algorithm * num_cases * (num_algorithm*num_cases + 1) * (2 * num_algorithm * num_cases + 1)) / 6
    denominator = denominator - (ranks_i / num_algorithm)
    stadistic_friedman = numerator / denominator

    ranks = [df[i].mean() for i in columns_names]
    rankings_with_label = {j: i for i, j in zip(ranks, columns_names)}

    if verbose:
        print(df_aligned)
        print(df[columns_names])
        print(stadistic_friedman)
        print(f"Rankings {ranks_j}")

    hypothesis_state = True
    if num_cases > 15 or num_algorithm >= 3:  # > 4
        # P-value = P(chi^2_{k-1} >= Q)
        p_value, critical_value = stats.get_p_value_chi2(stadistic_friedman, num_algorithm-1, alpha=alpha)
        if p_value < alpha:
            hypothesis_state = False
    else:
        p_value, critical_value = None, stats.get_cv_q_distribution(num_cases, num_algorithm - 1, alpha)
        if stadistic_friedman <= critical_value:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = False

    hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    if hypothesis_state:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

    return rankings_with_label, stadistic_friedman, p_value, critical_value, hypothesis


def quade(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False):
    """
    Perform the Quade test, a non-parametric statistical test used to identify significant differences between
    three or more matched groups. This test is particularly useful for blocked designs where treatments are
    applied to matched groups or blocks. The Quade test considers the relative differences between treatments
    within each block and ranks these differences.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A DataFrame with the first column as the block or subject identifier and the remaining
        columns as different treatments or conditions.
    alpha : float, optional
        The significance level for the test, default is 0.05.
    minimize : bool, optional
        Determines whether the metric of interest should be minimized or maximized. If True, the metric is to be
        minimized; if False, it is to be maximized. This parameter is crucial for tailoring the function's behavior to
        the specific nature of the data or the goal of the analysis.
    verbose : bool, optional
        If True, prints the detailed results table including ranks.

    Returns
    -------
    rankings_with_label : dict
        A dictionary with the average ranks of each treatment or condition.
    statistic_quade : float
        The Quade test statistic.
    p_value : float
        The p-value for the hypothesis test.
    cv_alpha_selected : None
        The critical value for the test at the specified alpha level.
    hypothesis : str
        A string stating the conclusion of the test based on the test statistic, critical value, and
        alpha.

    Note
    ----
    The Quade test adjusts for differences in treatment effects within each block by incorporating the range
    of each block into the ranking process. This makes it more sensitive to treatment effects in the presence of
    block-to-block variability. It's appropriate for ordinal data and assumes that the treatments are independent
    and identically distributed within each block.
    """
    columns_names = list(dataset.columns)
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    df = dataset.copy()

    columns_names.pop(0)
    df[columns_names] = df[columns_names].astype(float)

    # Compute the range of each block
    max_results = dataset[columns_names].max(axis=1)
    min_results = dataset[columns_names].min(axis=1)
    sample_range = max_results - min_results
    df["sample_range"] = sample_range
    sample_range_list = sample_range.values.flatten().tolist()
    aligned_observations = sorted(sample_range_list)
    ranking_cases = [aligned_observations.index(v) + 1 + (aligned_observations.count(v) - 1) / 2. for v in
                     sample_range_list]
    df["Rank_Q_i"] = ranking_cases

    # Assign rankings to each treatment within each block
    for index in df.index:
        row_values = dataset.loc[index][columns_names].values.flatten().tolist()
        # Sort depending on whether we minimize or maximize
        row_sort = sorted(row_values, reverse=not minimize)
        for alg in columns_names:
            v = df.loc[index][alg]
            # Assign rank, accounting for ties
            df.at[index, alg] = row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2.

    # Compute:
    # 1. relative_size: matrix S with S_ij = Rank_Q_i * (r_ij - (k+1)/2)
    # 2. rankings_without_average_adjusting: weighted values to compute average ranks per treatment
    relative_size = []
    rankings_without_average_adjusting = []

    for index in df.index:
        # Rankings are already assigned in the DataFrame
        row = np.array([df.at[index, alg] for alg in columns_names])
        r = df.at[index, "Rank_Q_i"]
        # Compute S for block i
        S_i = [r * (value - (num_algorithm + 1) / 2.) for value in row]
        relative_size.append(S_i)
        # Weighted values for computing average ranks
        rankings_without_average_adjusting.append(list(r * row))

    # Compute unnormalized average ranking for each treatment
    rankings_without_average_adjusting_to_algorithm = [
        sum(row[j] for row in rankings_without_average_adjusting) for j in range(num_algorithm)
    ]
    rankings_avg = [w / (num_cases * (num_cases + 1) / 2.) for w in rankings_without_average_adjusting_to_algorithm]

    # Compute statistics A and B based on the S matrix
    A = sum(sum(x ** 2 for x in row) for row in relative_size)
    Sj = [sum(relative_size[i][j] for i in range(num_cases)) for j in range(num_algorithm)]
    B = sum(s ** 2 for s in Sj) / float(num_cases)

    # Compute the Quade F-statistic and p-value using the F distribution
    tolerance = 1e-10
    if abs(A - B) < tolerance:
        stadistic_quade = 0
        p_value = 1.0
        critical_value = None
    else:
        stadistic_quade = (num_cases - 1) * B / (A - B)
        p_value = stats.get_p_value_f(stadistic_quade, num_algorithm - 1, (num_algorithm - 1) * (num_cases - 1))
        critical_value = stats.get_cv_f_distribution(num_algorithm - 1, (num_algorithm - 1) * (num_cases - 1), alpha)

    # Interpret the result of the test
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"
    else:
        hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"

    if verbose:
        print(df)

    # Build a dictionary with average ranks as keys and algorithm names as values
    rankings_with_label = {alg: rank for rank, alg in zip(rankings_avg, columns_names)}

    return rankings_with_label, stadistic_quade, p_value, critical_value, hypothesis


def kruskal_wallis(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False):
    """
    Perform the Kruskal-Wallis H-test for independent samples to determine whether the population
    medians on a dependent variable are the same across all groups or not. This non-parametric method
    does not assume a normal distribution of the residuals, making it a robust alternative to the one-way ANOVA.

    The function includes an option to adjust for ties and provides detailed output, including the test statistic,
    p-value, and a critical value based on the significance level (alpha). It can also print the dataset if verbose
    mode is enabled.

    Parameters
    ----------
    dataset : pd.DataFrame
        A pandas DataFrame containing the dataset to be tested. The first column should be the grouping variable,
        and the remaining columns should contain the data for each group.
    alpha : float, optional
        Significance level used to determine the critical value for the test. Default is 0.05.
    minimize : bool, optional
        Determines whether the metric of interest should be minimized or maximized. If True, the metric is to be
        minimized; if False, it is to be maximized. This parameter is crucial for tailoring the function's behavior to
        the specific nature of the data or the goal of the analysis.
    verbose : bool, optional
        If True, prints the dataset after melting and ranking. Useful for debugging or detailed analysis. Default is
        False.

    Returns
    -------
    statistic_kruskal : float
        The Kruskal-Wallis H statistic, adjusted for ties.
    p_value : float
        The p-value for the test statistic.
    critical_value : float
        The critical chi-square value from the chi-square distribution table based on the alpha level.
    hypothesis : str
        A string stating the result of the hypothesis test based on the p-value and alpha.

    Notes
    -----
    - The test is applied to data that is at least ordinal (rankable).
    - This implementation automatically adjusts for ties in the data.
    - The function converts the input data into long format, ranking the data, and then computes the test statistic.
    """

    def adjust_for_ties(rank_values):
        # Sort the rank values to identify ties
        sorted_values = np.sort(rank_values)

        # Identify the indices where values change (start and end of ties)
        value_changes_idx = np.nonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])[0]

        # Calculate the size of each tie group
        tie_sizes = np.diff(value_changes_idx).astype(np.float64)

        # Calculate the total sample size
        sample_size = np.float64(rank_values.size)

        # Calculate the correction factor for ties
        correction_factor = 1.0 if sample_size < 2 else 1.0 - (tie_sizes ** 3 - tie_sizes).sum() / (
                sample_size ** 3 - sample_size)

        return correction_factor

    data = pd.melt(dataset, id_vars=dataset.columns[0], var_name='Method', value_name='Performance')
    dv, between = 'Performance', 'Method'
    # Extract number of groups and total sample size
    n_groups = data[between].nunique()
    n = data[dv].size

    # Rank data, dealing with ties appropriately
    data["rank"] = data[dv].rank(method="average", ascending=minimize)

    # Find the total of rank per group
    grp = data.groupby(between, observed=True)["rank"]
    sum_rank_group = grp.sum().to_numpy()
    n_per_group = grp.count().to_numpy()

    # Calculate chi-square statistic (H)
    statistic_kruskal = (12 / (n * (n + 1)) * np.sum(sum_rank_group ** 2 / n_per_group)) - 3 * (n + 1)

    # Correct for ties
    correction = adjust_for_ties(data["rank"].to_numpy())
    statistic_kruskal /= correction

    # Calculate degrees of freedom and p-value
    degrees_of_freedom = n_groups - 1
    p_value, critical_value = stats.get_p_value_chi2(statistic_kruskal, degrees_of_freedom, alpha=alpha)

    if verbose:
        print(data)

    hypothesis = f"Fail to Reject H0 with alpha = {alpha} (Same distributions)"
    if p_value < alpha:
        hypothesis = f"Reject H0 with alpha = {alpha} (Different distributions)"

    return statistic_kruskal, p_value, critical_value, hypothesis


# -------------------- Test Multiple Groups -------------------- #


# Nemenyi

# -------------------- Post-Hoc Test -------------------- #
def nemenyi(ranks: dict, num_cases: int, alpha: float = 0.05, verbose: bool = False):
    """
    The Nemenyi test is a post-hoc analysis method used in statistics to compare multiple algorithms or treatments.
    It is often used after a Friedman test has indicated significant differences across algorithms.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings
        were calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05,
        which is a common choice in statistical testing.
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance).
        Defaults to False.

    Returns
    -------
    rank_values : list
        The list of rank values for each algorithm.
    critical_distance : float
        The critical distance for the Nemenyi test, which is a threshold used to determine if differences
        between algorithm rankings are statistically significant.
    figure : matplotlib.figure.Figure
        A figure representing the ranking of the algorithms and the critical distances visually,
        often as a CD diagram.

    Note
    ----
    The Nemenyi test is non-parametric and is used when the assumptions of parametric tests (like ANOVA)
    are not met. It's particularly useful in scenarios where multiple algorithms are compared across various datasets.
    """

    def graph_ranks(avg_ranks, names, cd=None, lowv=None, highv=None, width: float = 6.0, textspace: float = 1.0,
                    reverse: bool = False):
        width = float(width)
        textspace = float(textspace)

        def nth(elements, index):
            index = lloc(elements, index)
            return [a[index] for a in elements]

        def lloc(elements, index):

            if index < 0:
                return len(elements[0]) + index
            else:
                return index

        def mxrange(lr):

            if not len(lr):
                yield ()
            else:
                index = lr[0]
                if isinstance(index, int):
                    index = [index]
                for a in range(*index):
                    for b in mxrange(lr[1:]):
                        yield tuple([a] + list(b))

        average_ranks_copy = avg_ranks

        rankings_sort = sorted([(a, i) for i, a in enumerate(average_ranks_copy)], reverse=reverse)
        subset_rankings = nth(rankings_sort, 0)
        sorted_ids = nth(rankings_sort, 1)

        names_ids = [names[x] for x in sorted_ids]

        if lowv is None:
            lowv = min(1, int(math.floor(min(subset_rankings))))
        if highv is None:
            highv = max(len(avg_ranks), int(math.ceil(max(subset_rankings))))

        # print(rankings_sort)
        cline = 0.4

        k = len(average_ranks_copy)

        lines = None

        linesblank = 0
        scalewidth = width - 2 * textspace

        def rankpos(rank):
            if not reverse:
                a = rank - lowv
            else:
                a = highv - rank
            return textspace + scalewidth / (highv - lowv) * a

        distanceh = 0.25

        if cd:
            # get pairs of non significant methods

            def get_lines(sums, hsd):
                # get all pairs
                lsums = len(sums)
                allpairs = [(i, j) for i, j in mxrange([[lsums], [lsums]]) if j > i]
                # remove not significant
                not_significant = [(i, j) for i, j in allpairs if abs(sums[i] - sums[j]) <= hsd]

                # keep only longest

                def no_longer(ij_tuple, not_significant_pairs):
                    i, j = ij_tuple
                    for i1, j1 in not_significant_pairs:
                        if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                            return False
                    return True

                longest = [(i, j) for i, j in not_significant if no_longer((i, j), not_significant)]

                return longest

            lines = get_lines(subset_rankings, cd)
            linesblank = 0.2 + 0.2 + (len(lines) - 1) * 0.1

            # add scale
            distanceh = 0.25
            cline += distanceh

        # calculate height needed height of an image
        minnotsignificant = max(2 * 0.2, linesblank)
        height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

        fig = plt.figure(figsize=(width + 5, height))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
        ax.set_axis_off()

        height_factor = 1. / height  # height factor
        width_factor = 1. / width

        def hfl(elements):
            return [a * height_factor for a in elements]

        def wfl(elements):
            return [a * width_factor for a in elements]

        # Upper left corner is (0,0).
        ax.plot([0, 1], [0, 1], alpha=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

        def line(points_to_draw, color='k', **kwargs):
            """
            Input is a list of pairs of points.
            """
            ax.plot(wfl(nth(points_to_draw, 0)), hfl(nth(points_to_draw, 1)), color=color, **kwargs)

        def text(x, y, s, *args, **kwargs):
            ax.text(width_factor * x, height_factor * y, s, *args, **kwargs)

        line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

        bigtick = 0.1
        smalltick = 0.05

        tick = None
        for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
            tick = smalltick
            if a == int(a):
                tick = bigtick
            line([(rankpos(a), cline - tick / 2),
                  (rankpos(a), cline)],
                 linewidth=0.7)

        for a in range(lowv, highv + 1):
            text(rankpos(a), cline - tick / 2 - 0.05, str(a),
                 ha="center", va="bottom")

        k = len(subset_rankings)

        for i in range(math.ceil(k / 2)):
            chei = cline + minnotsignificant + i * 0.2
            line([(rankpos(subset_rankings[i]), cline),
                  (rankpos(subset_rankings[i]), chei),
                  (textspace - 0.1, chei)],
                 linewidth=0.7)
            text(textspace - 0.2, chei, f"{names_ids[i]} ({round(rankings_sort[i][0], 4)})", ha="right", va="center")

        for i in range(math.ceil(k / 2), k):
            chei = cline + minnotsignificant + (k - i - 1) * 0.2
            line([(rankpos(subset_rankings[i]), cline),
                  (rankpos(subset_rankings[i]), chei),
                  (textspace + scalewidth + 0.1, chei)],
                 linewidth=0.7)
            text(textspace + scalewidth + 0.2, chei, f"{names_ids[i]} ({round(rankings_sort[i][0], 4)})",
                 ha="left", va="center")

        if cd:
            # upper scale
            if not reverse:
                begin, end = rankpos(lowv), rankpos(lowv + cd)
            else:
                begin, end = rankpos(highv), rankpos(highv - cd)

            line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
            line([(begin, distanceh + bigtick / 2),
                  (begin, distanceh - bigtick / 2)],
                 linewidth=0.7)
            line([(end, distanceh + bigtick / 2),
                  (end, distanceh - bigtick / 2)],
                 linewidth=0.7)
            text((begin + end) / 2, distanceh - 0.05, f"CD ({round(cd, 4)})",
                 ha="center", va="bottom")

            # no-significance lines
            def draw_lines_cd(lines_to_draw, side=0.05, height_between=0.1):
                start = cline + 0.2
                for left_point, right_point in lines_to_draw:
                    line([(rankpos(subset_rankings[left_point]) - side, start),
                          (rankpos(subset_rankings[right_point]) + side, start)],
                         linewidth=2.5)
                    start += height_between

            draw_lines_cd(lines)

        return plt.gcf()

    names_algoriths = list(ranks.keys())
    num_algorithm = len(names_algoriths)
    q_alpha = stats.get_q_alpha_nemenyi(num_algorithm, alpha)

    critical_distance_nemenyi = q_alpha * math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases))
    if verbose:
        print(f"Distancia Crítica Nemenyi {critical_distance_nemenyi}")

    ranks_values = [ranks[i] for i in ranks.keys()]
    figure = graph_ranks(ranks_values, names_algoriths, cd=critical_distance_nemenyi, width=10, textspace=1.5)

    return ranks_values, critical_distance_nemenyi, figure


def _calculate_z_friedman(rank_i: float, rank_j: float, num_algorithm: int, num_cases: int):
    return abs(rank_i - rank_j) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))


def _calculate_z_friedman_aling(rank_i: float, rank_j: float, num_algorithm: int, num_cases: int):
    # return abs(rank_i - rank_j) / (math.sqrt((num_algorithm * (num_cases + 1)) / 6))
    return abs(rank_i - rank_j) / (math.sqrt((num_algorithm * (num_cases * num_algorithm + 1)) / 6))


def _calculate_z_quade(rank_i: float, rank_j: float, num_algorithm: int, num_cases: int):
    return abs(rank_i - rank_j) / (math.sqrt((num_algorithm * (num_algorithm + 1) * (2*num_cases + 1) *
                                              (num_algorithm - 1)) / (18 * num_cases * (num_cases + 1))))


def generate_graph_p_values(data: pd.DataFrame, name_control, all_vs_all):
    columns = ["Comparison", "p-value", "Adjusted alpha"]
    content = data[columns].to_numpy()

    if all_vs_all is False:
        for i in range(len(content)):
            content[i][0] = content[i][0][content[i][0].find("vs") + 3:]
    else:
        plt.tick_params(axis='x', labelrotation=45)

    content = list(content)
    content.append([name_control, 0.0, 0.0])

    content = sorted(content, key=lambda x: x[1])

    list_comparisons, list_p_values, thresholds = zip(*content)

    thresholds = [thresholds[0]] + list(thresholds) + [thresholds[-1]]

    possitions = [i - 0.5 for i in range(len(thresholds))]

    plt.grid(axis='y')

    # Crear el gráfico de barras
    plt.bar(range(len(list_p_values)), list_p_values, color="grey", edgecolor="black", label='p-values')

    # Personalizar las etiquetas del eje x
    plt.xticks(range(len(list_p_values)), list_comparisons)

    for i, value in enumerate(list_p_values):
        plt.text(i, value, f'{round(value, 5)}', ha='center', va='bottom')

    plt.step(possitions, thresholds, label='Thresholds', linestyle='--', color='black')

    plt.xlim(-0.5, len(list_p_values) - 0.5)
    # plt.tight_layout()
    plt.legend()
    # Mostrar el gráfico
    return plt.gcf()


def prepare_comparisons(ranks: dict, num_algorithm: int, num_cases: int, control: str = None,
                        type_rank: str = "Friedman"):
    all_vs_all = control is None
    algorithm_names = list(ranks.keys())
    index_control = 0 if all_vs_all else algorithm_names.index(control)
    ranks_values = [ranks[i] for i in ranks.keys()]
    available_ranks = {"Friedman": _calculate_z_friedman,
                       "Friedman + Iman Davenport": _calculate_z_friedman,
                       "Friedman Aligned Ranks": _calculate_z_friedman_aling,
                       "Quade": _calculate_z_quade
                       }
    results = []

    if not (type_rank in available_ranks.keys()):
        print(f"Error: Test of Rankings not available, stop functions")
        return -1

    calculate_z = available_ranks[type_rank]

    if all_vs_all:
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
                z_bonferroni = calculate_z(ranks_values[i], ranks_values[j], num_algorithm, num_cases)
                p_value = 2 * stats.get_p_value_normal(z_bonferroni)
                results.append((comparisons, z_bonferroni, p_value))
    else:
        for i in range(len(algorithm_names)):
            if index_control != i:
                comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
                z_bonferroni = calculate_z(ranks_values[index_control], ranks_values[i], num_algorithm, num_cases)
                p_value = 2 * stats.get_p_value_normal(z_bonferroni)
                results.append((comparisons, z_bonferroni, p_value))

    return results


def create_dataframe_results(comparisons: list, z_statistics: list, p_values: list, alphas: list, adj_p_values: list,
                             adj_alphas: list):

    results_h0 = ["Fail to Reject H0" if p_value > alpha else "Reject H0" for p_value, alpha in zip(p_values,
                                                                                                    adj_alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": z_statistics, "p-value": p_values,
                            "Adjusted alpha": adj_alphas, "Adjusted p-value": adj_p_values, "alpha": alphas,
                            "Results": results_h0})

    return results


def bonferroni(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
               type_rank: str = "Friedman", verbose: bool = False):
    """
    This function performs the Bonferroni correction for multiple comparisons of algorithms or treatments.
    The Bonferroni correction is a method to adjust significance levels when multiple statistical tests
    are conducted simultaneously.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were
        calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common
        choice in statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None,
        all algorithms are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults
        to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    This function is useful in statistical analysis where multiple algorithms or treatments are compared
    and there is a need to control the Type I error (false positive) that increases with the number of comparisons.
    """
    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    
    # num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if control is None else num_algorithm - 1
    num_of_comparisons = len(comparisons)
    
    # Adjusted alpha and p_values
    alpha_bonferroni = alpha / num_of_comparisons

    adj_alphas = [alpha_bonferroni] * len(comparisons)
    adj_p_values = [min((num_of_comparisons * p_value), 1) for p_value in p_values]
    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def holm(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
         type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Holm-Bonferroni method, an adjustment for multiple comparisons.
    It is used to control the family-wise error rate when comparing multiple algorithms or treatments.
    This method is more powerful than the simple Bonferroni correction, especially when the number
    of comparisons is large.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The ranks must be
        obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common choice in
        statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None, all algorithms
        are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Holm-Bonferroni method adjusts the significance levels of each comparison based on the order of
    their p-values, providing a less conservative approach than the original Bonferroni correction. It's particularly
    useful in scenarios with multiple comparisons to control the overall type I error rate.
    """

    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    
    # num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if control is None else num_algorithm - 1
    num_of_comparisons = len(comparisons)
    
    # Adjusted alpha and p_values
    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_alphas = [(alpha / (num_of_comparisons - index), value[0]) for index, value in enumerate(p_values_with_index)]
    adj_alphas = sorted(adj_alphas, key=lambda x: x[1])

    adj_alphas, _ = zip(*adj_alphas)

    adj_p_values = [max(((num_of_comparisons - j) * p_values_with_index[j][1], p_values_with_index[j][0]) for j in
                        range(i+1)) for i in range(num_of_comparisons)]

    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])

    adj_p_values, _ = zip(*adj_p_values)
    adj_p_values = [min(i, 1) for i in adj_p_values]

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def holland(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
            type_rank: str = "Friedman", verbose: bool = False):
    """
    This function applies the Holland step-down procedure for controlling the family-wise error rate in multiple
    comparisons. It's an improvement over the simple Bonferroni method and is particularly useful when dealing with a
    large number of comparisons, and it is generally more powerful than the Bonferroni correction.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The ranks must be
        obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common choice in
        statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None, all algorithms
        are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Holland method adjusts the alpha values for each comparison based on their rank in p-value, offering a more
    nuanced control over Type I errors compared to the Bonferroni method. It's particularly effective in scenarios with
    multiple algorithm comparisons, ensuring a more accurate interpretation of statistical results.
    """
    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    
    # num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if control is None else num_algorithm - 1
    num_of_comparisons = len(comparisons)
    
    # Adjusted alpha and p_values
    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])
    # Creo que debería de ser en vez de (num_algorithm - 1) el número de comparaciones
    adj_alphas = [(1 - (1-alpha)**(num_of_comparisons - index), value[0]) for index, value in
                  enumerate(p_values_with_index)]
    adj_alphas = sorted(adj_alphas, key=lambda x: x[1])

    adj_alphas, _ = zip(*adj_alphas)

    # Creo que debería de ser en vez de (num_algorithm ) el número de comparaciones + 1
    adj_p_values = [max([(1 - (1 - p_values_with_index[j][1])**(num_of_comparisons-j), p_values_with_index[j][0]) for j
                         in range(i+1)], key=lambda x: x[0]) for i in range(num_of_comparisons)]

    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])

    adj_p_values, _ = zip(*adj_p_values)
    adj_p_values = [min(i, 1) for i in adj_p_values]

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def finner(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
           type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Finner correction, a statistical method for controlling the family-wise error rate
    in multiple comparisons. The Finner correction is an alternative to other methods like Bonferroni or Holm,
    and it is generally more powerful than the Bonferroni correction.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were
        calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common
        choice in statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None,
        all algorithms are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance).
        Defaults to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Finner method is particularly useful in scenarios involving multiple comparisons, where it provides
    a balance between statistical power and control over Type I errors. This makes it a valuable tool in comparative
    studies of algorithms or treatments across various datasets.
    """
    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    
    # num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if control is None else num_algorithm - 1
    num_of_comparisons = len(comparisons)
    
    # Adjusted alpha and p_values
    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_alphas = [(1 - (1-alpha)**(num_of_comparisons/float(index+1)), value[0]) for index, value in
                  enumerate(p_values_with_index)]
    adj_alphas = sorted(adj_alphas, key=lambda x: x[1])
    adj_alphas, _ = zip(*adj_alphas)

    adj_p_values = [max([(1 - (1 - p_values_with_index[j][1])**(num_of_comparisons/float(j+1)),
                          p_values_with_index[j][0]) for j in range(i+1)], key=lambda x: x[0]) for i in
                    range(num_of_comparisons)]
    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)
    adj_p_values = [min(i, 1) for i in adj_p_values]

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def hommel(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
           type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Hommel correction, a statistical method for adjusting p-values when
    performing multiple comparisons. The Hommel correction is a more powerful alternative to the Bonferroni
    correction, particularly when the number of comparisons is large.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The ranks must be
        obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common choice in
        statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None, all algorithms
        are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Hommel correction is particularly useful in scenarios with multiple comparisons, as it provides a balance
    between controlling the family-wise error rate and maintaining statistical power. This makes it an effective tool
    in the comparative analysis of algorithms or treatments across various datasets.
    """
    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    
    # num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if control is None else num_algorithm - 1
    num_of_comparisons = len(comparisons)
    
    # Adjusted alpha and p_values
    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_alphas = [(alpha / (num_of_comparisons - index), value[0]) for index, value in enumerate(p_values_with_index)]
    adj_alphas = sorted(adj_alphas, key=lambda x: x[1])

    adj_alphas, _ = zip(*adj_alphas)
    
    length = len(p_values)
    adj_p_values = list(p_values).copy()
    c_list = [0.0] * length

    indices_sorted = sorted(range(len(p_values)), key=lambda x: p_values[x])

    for m in range(length, 1, -1):
        upper_range = list(range(length, length - m, -1))
        for i in upper_range:
            c_list[i-1] = (m * p_values[indices_sorted[i-1]]) / (m + i - length)
        
        c_min = min(c_list[i-1] for i in upper_range)
        
        for i in upper_range:
            adj_p_values[indices_sorted[i-1]] = max(adj_p_values[indices_sorted[i-1]], c_min)
        
        for i in range(length - m):
            c_list[i] = min(c_min, m * p_values[indices_sorted[i]])
            adj_p_values[indices_sorted[i]] = max(adj_p_values[indices_sorted[i]], c_list[i])

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def rom(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
        type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Rom method, which is a step-down procedure for adjusting p-values in the context of
    multiple comparisons. The Rom method is an improvement over the Bonferroni correction, providing a more powerful
    approach, especially when the number of comparisons is large.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were
        calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common
        choice in statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None,
        all algorithms are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults
        to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : object
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Rom method is particularly useful in scenarios involving multiple comparisons, as it offers a more
    accurate control of the family-wise error rate compared to simpler methods like Bonferroni, especially in
    cases with a large number of algorithms or treatments being compared.
    """
    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    num_of_comparisons = len(comparisons)
    # Adjusted alpha and p_values
    length = len(p_values)
    adj_alphas = [0.0] * length
    adj_p_values = [0.0] * length

    adj_alphas[length-1], adj_alphas[length-2] = alpha, alpha / 2.0 
    adj_p_values[length - 1], adj_p_values[length - 2] = 1, 2

    for i in range(3, length + 1):
        sum_factor_1 = sum(alpha ** j for j in range(1, i-1))
        sum_factor_2 = sum(math.comb(i, j) * (adj_alphas[(length - 2) - j] ** (i-j)) for j in range(1, i-2))
        adj_alphas[length - i] = (sum_factor_1 - sum_factor_2) / float(i)
        adj_p_values[length - i] = adj_alphas[length - 1] / adj_alphas[length - i]

    p_values_with_index = list(enumerate(p_values))
    # El cambio realizado es invertir el orden de los p-valores para que el mayor tenga el menor de los pesos
    p_values_with_index = list(reversed(sorted(p_values_with_index, key=lambda x: x[1])))
    adj_p_values = [[min(max(adj_p_values[num_of_comparisons-j] * p_values_with_index[j - 1][1] for j in
                             range(num_of_comparisons, i, -1)), 1), p_values_with_index[i][0]] for i in
                    range(num_of_comparisons)]

    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)
    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def li(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, 
       type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Li method for adjusting p-values in multiple comparisons.
    The Li method is particularly useful when there is a control algorithm to compare
    against other algorithms. It adjusts p-values based on the performance of the control
    algorithm relative to others.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The ranks must be
        obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common choice in
        statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None, all algorithms
        are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : object
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Li method is effective in scenarios where a specific algorithm (the control) is of particular
    interest, and the comparisons are made between this control and other algorithms. It provides a nuanced
    way to adjust for multiple comparisons by considering the performance of the control algorithm.
    """

    algorithm_names = list(ranks.keys())
    num_algorithm = len(ranks.keys())
    if control is None or control not in ranks.keys():
        print(f"Warning: Control algorithm don't found, we continue with best ranks")
        control = list(ranks.keys())[0]

    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)

    comparisons, z_bonferroni, p_values = zip(*results_comp)

    # Adjusted alpha and p_values
    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_p_values = [(p_values_with_index[i][1] / (p_values_with_index[i][1] + 1 - p_values_with_index[-1][1]),
                     p_values_with_index[i][0]) for i in range(len(p_values))]
    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)
    
    adj_alphas = [((1 - p_values_with_index[-1][1]) / (1 - alpha)) * alpha] * len(p_values)

    alphas = [alpha] * len(adj_alphas)
    # Create Struct
    results = create_dataframe_results(comparisons, z_bonferroni, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    # results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)

    return results, figure


def hochberg(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None,
             type_rank: str = "Friedman", verbose: bool = False):
    """
    This function implements the Hochberg step-up procedure for multiple comparisons correction.
    It is an alternative to the Bonferroni method and is generally more powerful, especially when
    there are a large number of comparisons. The Hochberg procedure controls the family-wise error rate
    more effectively than some other methods.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were
        calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common
        choice in statistical testing.
    control : str, optional
        An optional string specifying a control algorithm against which others will be compared. If None,
        all algorithms are compared against each other.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults
        to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : object
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    The Hochberg procedure is particularly valuable when conducting multiple comparisons in statistical analysis,
    as it provides a more refined approach to controlling the Type I error rate compared to more conservative methods.
    """

    num_algorithm = len(ranks.keys())
    algorithm_names = list(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, control, type_rank)
    comparisons, statistic_z, p_values = zip(*results_comp)

    num_of_comparisons = len(comparisons)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    alphas = [alpha] * len(p_values)

    adj_alphas = [(alpha * (index + 1) / num_of_comparisons, value[0]) for index, value in
                  enumerate(p_values_with_index)]

    adj_alphas = sorted(adj_alphas, key=lambda x: x[1])
    adj_alphas, _ = zip(*adj_alphas)

    adj_p_values = [[min(max((num_of_comparisons + 1 - j) * p_values_with_index[j-1][1] for j in
                             range(num_of_comparisons, i, -1)), 1), p_values_with_index[i][0]] for i in
                    range(num_of_comparisons)]

    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)
    results = create_dataframe_results(comparisons, statistic_z, p_values, alphas, adj_p_values, adj_alphas)

    if verbose:
        print(results)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, control is None)
    return results, figure


def shaffer(ranks: dict, num_cases: int, alpha: float = 0.05, type_rank: str = "Friedman", verbose: bool = False):
    """
    This function applies Shaffer's multiple comparison procedure, which is a more refined method for adjusting
    p-values in the context of multiple comparisons. Shaffer's method is an extension of the Bonferroni correction
    and is generally more powerful, particularly when the number of comparisons is large.

    Parameters
    ----------
    ranks : dict
        A dictionary where keys are the names of the algorithms and values are their respective ranks. The
        ranks must be obtained based on multiple non-parametric tests.
    num_cases : int
        An integer representing the number of cases, datasets, or instances over which the rankings were
        calculated.
    alpha : float, optional
        A float representing the significance level used in the test. It defaults to 0.05, which is a common
        choice in statistical testing.
    type_rank : str, optional
        A string indicating the type of ranking used (e.g., "Friedman"). Defaults to "Friedman".
    verbose : bool, optional
        A boolean indicating whether to print additional information (like the critical distance). Defaults
        to False.

    Returns
    -------
    comparison_results : pandas.DataFrame
        A DataFrame with the results of the comparisons, including adjusted z-values and adjusted p-values.
    comparison_figure : matplotlib.figure.Figure
        A figure graphically displaying the results of the tests, typically a bar chart or scatter plot.

    Note
    ----
    Shaffer's method is particularly effective in scenarios with multiple comparisons, as it provides a more
    accurate control of the family-wise error rate compared to simpler methods like Bonferroni, especially
    when the number of algorithms or treatments being compared is large.
    """
    def _calculate_independent_tests(num: int):
        if num == 0 or num == 1:
            return {0}
        else:
            result = set()
            for j in reversed(range(1, num + 1)):
                tmp = _calculate_independent_tests(num - j)
                for s in tmp:
                    result = result.union({stats.binomial_coef(j, 2) + s})
            return list(result)

    k = len(ranks)
    algorithm_names = list(ranks.keys())
 
    m = int(k * (k - 1) / 2.)
    independent_test_hypotheses = _calculate_independent_tests(int((1 + math.sqrt(1 + 4 * m * 2)) / 2))
    t = [max([a for a in independent_test_hypotheses if a <= m - i]) for i in range(m)]
    
    num_algorithm = len(ranks.keys())
    results_comp = prepare_comparisons(ranks, num_algorithm, num_cases, None, type_rank)    

    comparisons, statistic_z, p_values = zip(*results_comp)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])
    adj_p_values = [(min(max(t[j] * p_values_with_index[j][1] for j in range(i + 1)), 1), p_values_with_index[i][0])
                    for i in range(m)]
    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)

    alphas = [alpha] * len(p_values)
    results = create_dataframe_results(comparisons, statistic_z, p_values, alphas, adj_p_values, alphas)

    control = algorithm_names[0]
    if verbose:
        print(results)

    figure = generate_graph_p_values(results, control, True)
    return results, figure

# -------------------- Post-Hoc Test -------------------- #


def mcnemar(results_1, results_2, real_results, alpha: float = 0.05, verbose: bool = False):
    if len(results_1) != len(results_2) or len(results_1) != len(real_results) or len(results_2) != len(real_results):
        print("Error: Todos los vectores proporcionados deben de tener el mismo número de elementos")
        print(f"results_1: {len(results_1)}, results_2: {len(results_2)}, real_results: {len(real_results)}")
        return -1

    data = {
        'Alg2_Error': [0, 0],
        'Alg2_OK': [0, 0]
    }

    index = ['Alg1_Error', 'Alg1_OK']

    matrix_mcnemar = pd.DataFrame(data, index=index)

    for i in range(len(results_1)):
        check_alg1 = int(results_1[i] == real_results[i])
        check_alg2 = int(results_2[i] == real_results[i])
        row = 'Alg1_OK' if check_alg1 else 'Alg1_Error'
        column = 'Alg2_OK' if check_alg2 else 'Alg2_Error'
        matrix_mcnemar.at[row, column] += 1

    if verbose:
        print(matrix_mcnemar)

    mcnemar_statistic = ((abs(matrix_mcnemar.at["Alg1_Error", "Alg2_OK"] - matrix_mcnemar.at["Alg1_OK", "Alg2_Error"])
                          - 1) ** 2) / (matrix_mcnemar.at["Alg1_Error", "Alg2_OK"] +
                                        matrix_mcnemar.at["Alg1_OK", "Alg2_Error"])

    p_value, cv_mcnemar = stats.get_p_value_chi2(mcnemar_statistic, 1, alpha)

    if verbose:
        print(f"McNemar statistic: {mcnemar_statistic}, CV McNemar with alpha {alpha}: {cv_mcnemar}")

    if p_value < alpha:
        print(f"Reject H0 with alpha = {alpha} (Different distributions)")
    else:
        print(f"Fail to Reject H0 with alpha = {alpha} (Same distributions)")


def multiple_sign_test(dataset: pd.DataFrame, alpha: float = 0.05, minimize: bool = False, verbose: bool = False):
    algorith_names = list(dataset.columns)[1:]
    columns_names_comparations = ["d1" + str(i) for i in range(2, len(algorith_names) + 1)]
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    distances = pd.DataFrame()
    control = dataset[algorith_names[0]]
    for i in columns_names_comparations:
        index = int(i[-1]) - 1
        current_alg = dataset[algorith_names[index]]
        dif = pd.DataFrame(control - current_alg, columns=[i])
        distances = pd.concat([distances, dif], axis=1)
        dataset[i] = distances[i].apply(lambda x: '+' if x >= 0 else '-')

    sign_to_count = "-" if not minimize else "+"
    r_count = [dataset[dataset[i] == sign_to_count].shape[0] for i in columns_names_comparations]

    table_cv = pd.read_csv("CV_Minimum_r.csv")
    columns_table_cv = list(table_cv.columns)
    cv_to_alpha = table_cv[(table_cv[columns_table_cv[0]] == num_cases) & (table_cv[columns_table_cv[1]] == alpha)]
    if cv_to_alpha.empty:
        return "Error: Tabla"

    cv_to_alpha = int(cv_to_alpha[f"k={num_algorithm-1}"])

    results = pd.DataFrame(columns=["Control", "Zi", "Ri", "Critical_Value", "Accept_H_0"])
    for i in range(len(r_count)):
        row_to_agregate = {"Control": [algorith_names[0]], "Zi": [algorith_names[i+1]], "Ri": [r_count[i]],
                           "Critical_Value": [cv_to_alpha], "Accept_H_0": [r_count[i] > cv_to_alpha]}
        df = pd.DataFrame(row_to_agregate)
        results = pd.concat([results, df])
    if verbose:
        print(dataset)
        print(results)

    return results


def contrast_estimation_based_on_medians(dataset: pd.DataFrame, minimize: bool = False, verbose: bool = False):
    #  Este test no proporciona probabilidad de error
    def calculate_mean_unajusted_medians(unadjusted_estimator: dict, index: int):
        def change_sign(string: str, value: str):
            if string[1] == value:
                return 1
            else:
                return -1
        a = [unadjusted_estimator[key] * change_sign(key, str(index + 1)) for key in unadjusted_estimator.keys()
             if str(index + 1) in key]

        return sum(a)

    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    name_algorithms = dataset.columns[1:]
    differences_between_pairs_alg = pd.DataFrame()
    for i in range(1, num_algorithm + 1):
        for j in range(i + 1, num_algorithm + 1):
            name_column = f"D{i}{j}"
            differences_between_pairs_alg[name_column] = dataset[name_algorithms[i-1]] - dataset[name_algorithms[j-1]]

    if verbose:
        print(differences_between_pairs_alg)
    unadjusted_estimators = {}
    for i in differences_between_pairs_alg.columns:
        unadjusted_estimators[i] = round(differences_between_pairs_alg[i].median(), 10)

    if verbose:
        print(unadjusted_estimators)
    constrast_estimation_results = pd.DataFrame(columns=name_algorithms, index=name_algorithms)

    for index_i, name_i in enumerate(name_algorithms):
        for index_j, name_j in enumerate(name_algorithms):
            estimate_dif = 0
            if index_i != index_j:
                # Se calcula el estimador
                m_i = calculate_mean_unajusted_medians(unadjusted_estimators, index_i)
                m_j = calculate_mean_unajusted_medians(unadjusted_estimators, index_j)

                estimate_dif = m_i - m_j

            constrast_estimation_results.at[name_i, name_j] = estimate_dif

    if not minimize:
        constrast_estimation_results *= -1

    if verbose:
        print(constrast_estimation_results)

    return constrast_estimation_results
