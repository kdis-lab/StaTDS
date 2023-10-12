import numpy as np
import math
import pandas as pd
import stats


def shapiro_wilk_normality(data: np.array, alpha: float = 0.05):

    sorted_data = np.sort(data)

    mean = sorted_data.mean()
    sum_square = np.sum((np.array(sorted_data) - mean) ** 2)

    num_samples = data.shape[0]
    m = num_samples // 2.0
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
    mean = data.mean()
    x_minus_mean = data - mean
    num_samples = data.shape[0]
    m_3 = np.sum(x_minus_mean ** 3) / num_samples
    m_2 = np.sum(x_minus_mean ** 2) / num_samples
    statistical_skew = m_3 / (math.sqrt(m_2 ** 3))

    if bias is False:
        statistical_skew = (math.sqrt(num_samples * (num_samples - 1)) / (num_samples - 2)) * statistical_skew

    return statistical_skew


def kurtosis(data: np.array, bias: bool = True):
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


def d_agostino_pearson(data: np.array, alpha: float = 0.05):
    sorted_data = np.sort(data)

    mean = sorted_data.mean()

    skew = skewness(sorted_data)
    kurt = kurtosis(sorted_data)
    num_samples = sorted_data.shape[0]

    ses = math.sqrt((6 * num_samples * (num_samples - 1)) / ((num_samples - 2) * (num_samples + 1) * (num_samples - 3)))
    sek = 2 * ses * math.sqrt((num_samples ** 2 - 1) / ((num_samples - 3) * (num_samples + 5)))

    standard_score_s = skew / ses
    standard_score_k = kurt / sek

    statistic_dp = standard_score_s ** 2 + standard_score_k ** 2

    # Calculate p-value with chi^2 with 2 degrees of freedom
    p_value, cv_value = stats.get_p_value_chi2(statistic_dp, 2, alpha=alpha)

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if p_value > alpha:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return statistic_dp, p_value, cv_value, hypothesis


def kolmogorov_smirnov(data: np.array, alpha: float = 0.05):
    def norm_cdf(x):
        # Función de distribución acumulativa (CDF) de una distribución normal estándar
        return [0.5 * (1 + math.erf((i - np.mean(x)) / (np.std(x) * np.sqrt(2)))) for i in x]

    sorted_data = np.sort(data)

    # Calcular la función de distribución acumulativa empírica (ECDF)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    # Calcular la diferencia máxima entre la ECDF y la CDF de una distribución normal
    d_max = np.max(np.abs(ecdf - norm_cdf(sorted_data)))

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

    return d_max, p_value, cv_value, hypothesis


def levene_test(dataset: pd.DataFrame, alpha: float = 0.05, center: str = 'mean'):
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
        calculate_center = np.mean  # TODO Change for the trimmed mean

    factor_inicial = (num_total - num_groups) / (num_groups - 1)

    center_of_groups = [calculate_center(dataset[i]) for i in names_groups]

    z_ij = [abs(dataset[names_groups[i]] - center_of_groups[i]).to_numpy() for i in range(len(names_groups))]

    z_bar_i = np.array([np.mean(i) for i in z_ij])
    z_bar = [i * j for i, j in zip(z_bar_i, num_samples)]
    z_bar = np.sum(z_bar)

    numer = np.sum(num_samples * (z_bar_i - z_bar) ** 2)
    dvar = np.sum([np.sum((z_ij[i] - z_bar_i[i]) ** 2, axis=0) for i in range(num_groups)])

    statistic_levene = factor_inicial * (numer / dvar)

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
    if dataset.shape[1] < 2:
        raise "Error: Bartlett Test need at least two samples"

    names_groups = list(dataset.columns)
    num_groups = len(names_groups)
    num_samples = [dataset[i].shape[0] for i in names_groups]
    num_total = np.sum(num_samples)

    std_groups = [np.std(dataset[i]) for i in names_groups]

    pooled_variance = np.sum([((num_samples[i] - 1) * (std_groups[i] ** 2)) / (num_total - num_groups) for i in
                              range(num_groups)])

    numerator = (num_total - num_groups) * math.log(pooled_variance ** 2) - (np.sum([(num_samples[i] - 1) *
                                                                                     math.log(std_groups[i] ** 2)
                                                                                     for i in range(num_groups)]))

    denominator = 1 + (1/(3.0 * (num_groups - 1))) * (np.sum([1 / (i - 1.0) for i in num_samples]) -
                                                      (1 / float(num_total - num_groups)))
    statistical_bartlett = numerator / denominator

    p_value, cv_value = stats.get_p_value_chi2(statistical_bartlett, num_groups-1, alpha=alpha)

    if p_value > alpha:
        print(f"Same distributions (fail to reject H0) with alpha {alpha}")
    else:
        print(f"Different distributions (reject H0) with alpha {alpha}")

    print("\t p_value = ", p_value)

    return statistical_bartlett, p_value, cv_value


def t_test_paired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
    if dataset.shape[1] != 2:
        raise "Error: T Test need two samples"

    names_groups = list(dataset.columns)
    num_samples = dataset.shape[0]
    mean_samples = np.mean(dataset[names_groups[0]] - dataset[names_groups[1]])
    std_samples = np.std(dataset[names_groups[0]] - dataset[names_groups[1]])

    standard_error_of_the_mean = std_samples / math.sqrt(num_samples)

    statistical_t = mean_samples / standard_error_of_the_mean

    # TODO Calculate P-Valor T Distribution with alpha, degrees_of_freedom = num_samples - 1 (Revisar tras hablar)

    # p_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    rejected_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    p_value = None

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_t < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)
    return statistical_t, rejected_value, p_value, hypothesis


def t_test_unpaired(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):
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
    # p_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    rejected_value = stats.get_cv_t_distribution(num_samples - 1, alpha=alpha)
    p_value = None
    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if statistical_t < rejected_value:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    if verbose is True:
        print(statistical_t, rejected_value, p_value, hypothesis)

    return statistical_t, rejected_value, p_value, hypothesis


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


# PRUEBAS UNITARIAS DE LAS FUNCIONES QUITAR ESTO DE AQUI MÁS ADELANTE


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
