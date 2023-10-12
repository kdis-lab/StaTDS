import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import re
import stats


class LibraryError(Exception):
    def __init__(self, message):
        super().__init__(message)


# -------------------- Test Two Groups -------------------- #
def wilconxon(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):

    if dataset.shape[1] != 2:
        raise "Error: The test only needs two samples"

    results_table = dataset.copy()
    columns = list(dataset.columns)
    differences_results = dataset[columns[0]] - dataset[columns[1]]
    absolute_dif = differences_results.abs()
    absolute_dif = absolute_dif.sort_values()
    results_wilconxon = {"index": [], "dif": [], "rank": [], "R": []}
    rank = 0.0
    # Esto se deberá de arreglar para cuando se repitan valores en las diferencias
    tied_ranges = not(len(set(absolute_dif)) == absolute_dif.shape[0])

    for index in absolute_dif.index:
        if math.fabs(0 - absolute_dif[index]) < 1e-10:
            # Se continua con el siguiente
            continue  # Debido a que la diferencia entre ambos algoritmos es 0 y no se debe de tener en cuenta.
        rank += 1.0
        results_wilconxon["index"] += [index]
        results_wilconxon["dif"] += [differences_results[index]]
        results_wilconxon["rank"] += [rank]
        results_wilconxon["R"] += ["+" if differences_results[index] > 0 else "-"]

    df = pd.DataFrame(results_wilconxon)
    df = df.set_index("index")
    df = df.sort_index()
    results_table = pd.concat([results_table, df], axis=1)

    if tied_ranges:
        # Realizar el calculo de los rangos teniendo en cuenta que se comparten valores
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

    if verbose:
        print(results_table)

    r_plus = results_table[results_table.R == "+"]["rank"].sum()
    r_minus = results_table[results_table.R == "-"]["rank"].sum()

    w_wilcoxon = min([r_plus, r_minus])
    num_problems = results_table.shape[0]
    mean_wilcoxon = (num_problems * (num_problems + 1)) / 4.0
    std_wilcoxon = math.sqrt((num_problems * (num_problems + 1) * ((2 * num_problems) + 1)) / 24.0)
    z_wilcoxon = (w_wilcoxon - mean_wilcoxon) / std_wilcoxon

    if num_problems > 25:
        # Se puede aproximar a una N(0,1)
        p_value = stats.get_p_value_normal(z_wilcoxon)

        # print(f"p_value {p_value}, Stadistic: {w_wilcoxon}, Z_wilcoxon {z_wilcoxon}")
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
        if p_value < alpha:
            hypothesis = f"Different distributions (reject H0) with alpha {alpha}"

        return w_wilcoxon, None, p_value, hypothesis

    else:
        # Se debe de utilizar las tablas con los valores críticos

        cv_alpha_selected = stats.get_cv_willcoxon(num_problems, alpha)

        # print(f"Critical Value with alpha {alpha} and {num_problems} problems -> {cv_alpha_selected}, stadistic: "
        #       f"{w_wilcoxon}")

        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
        if cv_alpha_selected < w_wilcoxon:
            hypothesis = f"Different distributions (reject H0) with alpha {alpha}"

        return w_wilcoxon, cv_alpha_selected, None, hypothesis


def binomial(dataset: pd.DataFrame, alpha: float = 0.05, verbose: bool = False):

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

    hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
    if p_value < alpha:
        hypothesis = f"Different distributions (reject H0) with alpha {alpha}"

    return statistical_binomial, None, p_value, hypothesis


def mannwhitneyu(dataset: pd.DataFrame, alpha: float = 0.05, criterion: bool = False, verbose: bool = False):

    if dataset.shape[1] != 2:
        raise "Error: The test only needs two samples"

    columns = list(dataset.columns)

    values_dataset = list(dataset.stack())

    aligned_observations = sorted(values_dataset)
    aligned_observations = list(reversed(aligned_observations)) if criterion else aligned_observations

    ranking_cases = [aligned_observations.index(v) + 1 + (aligned_observations.count(v) - 1) / 2. for v
                     in values_dataset]

    rank_alg1, rank_alg2 = zip(*[(ranking_cases[i], ranking_cases[i + 1]) for i in range(0, len(ranking_cases), 2)])

    ranks = pd.DataFrame({"Ranks_"+columns[0]: rank_alg1, "Ranks_"+columns[1]: rank_alg2})
    dataset = pd.concat([dataset, ranks], axis=1)

    rank_alg1, rank_alg2 = ranks.mean()

    if verbose:
        print(dataset)
        print("Ranks_"+columns[0] + ": ", rank_alg1, "Ranks_"+columns[1] + ": ", rank_alg2)

    num_cases = dataset.shape[0]

    statistics_u_alg1 = num_cases * num_cases + (num_cases * (num_cases + 1)) / 2.0 - rank_alg1
    statistics_u_alg2 = num_cases * num_cases + (num_cases * (num_cases + 1)) / 2.0 - rank_alg2

    statistical_u = max(statistics_u_alg1, statistics_u_alg2)

    mean_u = (num_cases * num_cases) / 2.0

    std_u = math.sqrt((num_cases * num_cases) * (num_cases + num_cases + 1) / 12.0)

    z_value = (statistical_u - mean_u) / std_u

    p_value = stats.get_p_value_normal(z_value)

    hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"
    if p_value < alpha:
        hypothesis = f"Different distributions (reject H0) with alpha {alpha}"

    return z_value, None, p_value, hypothesis

# -------------------- Test Two Groups -------------------- #


# -------------------- Test Multiple Groups -------------------- #
def friedman(dataset: pd.DataFrame, alpha: float = 0.05, criterion: bool = False, verbose: bool = False):
    columns_names = list(dataset.columns)
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    ranks = []
    for i in dataset.index:
        sr = dataset.loc[i][columns_names[1:]]
        ranks.append(sr.rank(method='average', ascending=(not criterion)).tolist())

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

    hypothesis_state = False
    if verbose:
        print(df)
    rankings_with_label = {j: i / num_cases for i, j in zip(ranks, columns_names[1:])}

    # Revisar esta parte
    if num_cases > 15 or num_algorithm > 4:
        # P-value = P(chi^2_{k-1} >= Q)
        # Cargamos la tabla estadística
        reject_value = stats.get_p_value_chi2(stadistic_friedman, num_algorithm-1, alpha)
        if stadistic_friedman < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = True
    else:
        # No se puede usar la chi^2, debemos de usar las tablas de la distribución Q
        reject_value = [None, stats.get_cv_q_distribution(num_cases, num_algorithm-1, alpha)]
        if stadistic_friedman < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = True
    # Interpret
    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if hypothesis_state:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return rankings_with_label, stadistic_friedman, reject_value[0], reject_value[1], hypothesis


def friedman_aligned_ranks(dataset: pd.DataFrame, alpha: float = 0.05, criterion: bool = False, verbose: bool = False):
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
    aligned_observations = list(reversed(aligned_observations)) if criterion else aligned_observations
    # criterion El criterio por defecto es minimizar
    df_aligned = df.copy()

    for index in df.index:
        for algorith in columns_names:
            v = df.loc[index][algorith]
            df.at[index, algorith] = aligned_observations.index(v) + 1 + (aligned_observations.count(v) - 1) / 2.

    ranks_j = [(df[i].sum()) ** 2 for i in columns_names]
    ranks_i = df[columns_names].sum(axis=1)
    ranks_i = (ranks_i ** 2).sum()

    stadistic_friedman = ((num_algorithm - 1) * (sum(ranks_j) - (num_algorithm * ((num_cases ** 2) / 4.0) *
                                                                 (num_algorithm * num_cases + 1) ** 2))) / (
                             float(((num_algorithm * num_cases * (num_algorithm * num_cases + 1) *
                                     (2 * num_algorithm * num_cases + 1)) / 6.) - (
                                               1. / float(num_algorithm)) * ranks_i))

    if verbose:
        #  Se podría concatenar estas columnas al dataframe anterior para tener en una sola todos los valores
        print(df_aligned)
        print(df[columns_names])
        print(stadistic_friedman)
        print(f"Rankings {ranks_j}")

    ranks = [df[i].mean() for i in columns_names]
    rankings_with_label = {j: i for i, j in zip(ranks, columns_names)}

    hypothesis_state = False

    if num_cases > 15 or num_algorithm > 4:
        # P-value = P(chi^2_{k-1} >= Q)
        # Cargamos la tabla estadística
        reject_value = stats.get_p_value_chi2(stadistic_friedman, num_algorithm, alpha)
        if stadistic_friedman < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = True
    else:
        # No se puede usar la chi^2, debemos de usar las tablas de la distribución Q
        reject_value = [None, stats.get_cv_q_distribution(num_cases, num_algorithm-1, alpha)]
        if stadistic_friedman < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
            hypothesis_state = True

    # Interpret
    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if hypothesis_state:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    return rankings_with_label, stadistic_friedman, reject_value[0], reject_value[1], hypothesis


def quade(dataset: pd.DataFrame, alpha: float = 0.05, criterion: bool = False, verbose: bool = False):
    columns_names = list(dataset.columns)
    num_cases, num_algorithm = dataset.shape
    num_algorithm -= 1
    df = dataset.copy()

    columns_names.pop(0)
    max_results = dataset[columns_names].max(axis=1)
    min_results = dataset[columns_names].min(axis=1)
    sample_range = max_results - min_results
    df["sample_range"] = sample_range
    sample_range = sample_range.values.flatten().tolist()
    aligned_observations = sorted(sample_range)
    ranking_cases = [aligned_observations.index(v) + 1 + (aligned_observations.count(v) - 1) / 2. for v in sample_range]

    df["Rank_Q_i"] = ranking_cases

    # Calculo de los rankings de forma eficiente
    for index in df.index:
        row = sorted(dataset.loc[index][columns_names].values.flatten().tolist())
        row_sort = list(reversed(row)) if criterion else row
        for algorith in columns_names:
            v = df.loc[index][algorith]
            df.at[index, algorith] = row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2.

    relative_size = []
    rankings_without_average_adjusting = []

    for index in df.index:
        row = np.array(df.loc[index][columns_names].values)
        r = df.loc[index]["Rank_Q_i"]
        relative_size.append(list(r * (row - (num_algorithm + 1)/2.)))
        rankings_without_average_adjusting.append(list(row * r))

    relative_size_to_algorithm = [sum(row[j] for row in relative_size) for j in range(num_algorithm)]
    rankings_without_average_adjusting_to_algorithm = [sum(row[j] for row in rankings_without_average_adjusting) for j
                                                       in range(num_algorithm)]

    rankings_avg = [w / (num_cases * (num_cases + 1) / 2.) for w in rankings_without_average_adjusting_to_algorithm]
    """
    rankings_cmp = [r / math.sqrt(num_algorithm * (num_algorithm + 1) * (2 * num_cases + 1) * (num_algorithm - 1) /
                                  (18. * num_cases * (num_cases + 1))) for r in rankings_avg]
    """

    stadistic_a = sum(relative_size[i][j] ** 2 for i in range(num_cases) for j in range(num_algorithm))
    stadistic_b = sum(s ** 2 for s in relative_size_to_algorithm) / float(num_cases)

    hypothesis_state = False
    stadistic_quade = None
    if stadistic_a - stadistic_b > 0.0000000001:
        stadistic_quade = (num_cases - 1) * stadistic_b / (stadistic_a - stadistic_b)

        if num_cases > 15 or num_algorithm > 4:
            # P-value = P(chi^2_{k-1} >= Q)
            # Cargamos la tabla estadística
            reject_value = stats.get_p_value_chi2(stadistic_quade, num_algorithm, alpha)
            if stadistic_quade < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
                hypothesis_state = True

        else:
            # No se puede usar la chi^2, debemos de usar las tablas de la distribución Q
            reject_value = [None, stats.get_cv_q_distribution(num_cases, num_algorithm-1, alpha)]
            if stadistic_quade < reject_value[1]:  # valueFriedman < p(alpha >= chi^2)
                hypothesis_state = True

    else:
        p_value = math.pow((1 / float(math.factorial(num_algorithm))), num_cases - 1)
        hypothesis_state = True if p_value >= alpha else False
        reject_value = [p_value, None]

    # Si A = B se considera un región critical en la distribución estadística, y se calcula el p-valor como (1/k!)^n-1

    hypothesis = f"Different distributions (reject H0) with alpha {alpha}"
    if hypothesis_state:
        hypothesis = f"Same distributions (fail to reject H0) with alpha {alpha}"

    if verbose:
        print(df)

    rankings_with_label = {j: i for i, j in zip(rankings_avg, columns_names)}

    return rankings_with_label, stadistic_quade, reject_value[0], reject_value[1], hypothesis
# -------------------- Test Multiple Groups -------------------- #


# Nemenyi
def graph_ranks(avg_ranks, names, cd=None, lowv=None, highv=None, width: float = 6.0, textspace: float = 1.0,
                reverse: bool = False):
    # TODO Se deberá de depurar esta función para la generación del gráfico
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

    fig = plt.figure(figsize=(width+5, height))
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
    ax.plot([0, 1], [0, 1], c="w")
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

    # if name_fig == "":
    #    plt.show()
    # else:
    #    plt.savefig(name_fig)

    return plt.gcf()


# -------------------- Post-Hoc Test -------------------- #
def nemenyi(ranks: dict, num_cases: int, alpha: float = 0.05, verbose: bool = False):
    names_algoriths = list(ranks.keys())
    num_algorithm = len(names_algoriths)
    q_alpha = stats.get_q_alpha_nemenyi(num_algorithm, alpha)

    critical_distance_nemenyi = q_alpha * math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases))
    if verbose:
        print(f"Distancia Crítica Nemenyi {critical_distance_nemenyi}")

    ranks_values = [ranks[i] for i in ranks.keys()]
    figure = graph_ranks(ranks_values, names_algoriths, cd=critical_distance_nemenyi, width=10, textspace=1.5)

    return ranks_values, critical_distance_nemenyi, figure


def bonferroni(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None,
               verbose: bool = False):
    algorithm_names = list(ranks.keys())
    num_algorithm = len(algorithm_names)
    if not(control is None) and control not in algorithm_names:
        print(f"Warning: Control algorithm don't found, we continue All VS All")
        control = None

    all_vs_all, index_control = (True, -1) if control is None else (False, algorithm_names.index(control))

    num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if all_vs_all else num_algorithm - 1
    alpha_bonferroni = alpha / num_of_comparisons
    ranks = [ranks[i] for i in ranks.keys()]

    results_comp = []

    if all_vs_all:
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
                z_bonferroni = (ranks[i] - ranks[j]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                p_value = stats.get_p_value_normal(z_bonferroni)
                results_comp.append((comparisons, z_bonferroni, p_value))
    else:
        for i in range(len(algorithm_names)):
            if index_control != i:
                comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
                z_bonferroni = (ranks[index_control] - ranks[i]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                p_value = stats.get_p_value_normal(z_bonferroni)
                results_comp.append((comparisons, z_bonferroni, p_value))

    comparisons, z_bonferroni, p_values = zip(*results_comp)
    results_h0 = ["H0 is accepted" if p_value > alpha_bonferroni else "H0 is rejected" for p_value in p_values]

    alphas = [alpha_bonferroni] * len(comparisons)

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": z_bonferroni, "Adjusted p-value": p_values,
                            "Adjusted alpha": alphas, "Results": results_h0})

    if verbose:
        print(results)

    results.reset_index(drop=True, inplace=True)

    if control is None:
        control = algorithm_names[0]
    print("hola")
    figure = generate_graph_p_values(results, control, all_vs_all)
    print("ADIOS")
    return results, figure


def generate_graph_p_values(data: pd.DataFrame, name_control, all_vs_all):
    content = data[["Comparison", "Adjusted p-value", "Adjusted alpha"]].to_numpy()

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
    # plt.figure(figsize=(num_alg, 5))

    plt.grid(axis='y')

    # Crear el gráfico de barras
    plt.bar(range(len(list_p_values)), list_p_values, color="grey", edgecolor="black", label='p-values')

    # Personalizar las etiquetas del eje x
    plt.xticks(range(len(list_p_values)), list_comparisons)

    for i, value in enumerate(list_p_values):
        plt.text(i, value, f'{round(value, 5)}', ha='center', va='bottom')

    plt.step(possitions, thresholds, label='Thresholds', linestyle='--', color='black')

    plt.xlim(-0.5, len(list_p_values) - 0.5)
    #plt.tight_layout()
    plt.legend()
    # Mostrar el gráfico
    return plt.gcf()


def holm(ranks: dict, num_cases: int, alpha: float = 0.05, control: str = None, verbose: bool = False):
    algorithm_names = list(ranks.keys())
    num_algorithm = len(algorithm_names)
    if not(control is None) and control not in algorithm_names:
        print(f"Warning: Control algorithm don't found, we continue All VS All")
        control = None

    all_vs_all, index_control = (True, 0) if control is None else (False, algorithm_names.index(control))

    num_of_comparisons = (num_algorithm * (num_algorithm - 1)) / 2.0 if all_vs_all else num_algorithm - 1
    ranks = [ranks[i] for i in ranks.keys()]
    
    results_comp = []

    if all_vs_all:
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
                statistic_z = (ranks[i] - ranks[j]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))
    else:
        for i in range(len(algorithm_names)):
            if index_control != i:
                comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
                statistic_z = (ranks[index_control] - ranks[i]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))

    comparisons, statistic_z, p_values = zip(*results_comp)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    alphas = [(alpha / (num_of_comparisons - index), value[0]) for index, value in enumerate(p_values_with_index)]
    alphas = sorted(alphas, key=lambda x: x[1])

    alphas, _ = zip(*alphas)

    results_h0 = ["H0 is accepted" if p_value > alpha else "H0 is rejected" for p_value, alpha in zip(p_values, alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": statistic_z, "Adjusted p-value": p_values,
                            "Adjusted alpha": alphas, "Results": results_h0})

    if verbose:
        print(results)

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, all_vs_all)
    return results, figure


def hochberg(ranks: dict, alpha: float = 0.05, control: str = None):
    algorithm_names = list(ranks.keys())
    if not(control is None) and control not in algorithm_names:
        print(f"Warning: Control algorithm don't found, we continue All VS All")
        control = None

    all_vs_all, index_control = (True, 0) if control is None else (False, algorithm_names.index(control))
    results_comp = []
    if all_vs_all:
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
                # statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]])
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))
    else:
        for i in range(len(algorithm_names)):
            if index_control != i:
                comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
                # statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]])
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))

    comparisons, statistic_z, p_values = zip(*results_comp)
    num_comparisons = len(comparisons)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    alphas = [(alpha * (index + 1) / num_comparisons, value[0]) for index, value in enumerate(p_values_with_index)]
    alphas = sorted(alphas, key=lambda x: x[1])

    alphas, _ = zip(*alphas)

    results_h0 = ["H0 is accepted" if p_value > alpha else "H0 is rejected" for p_value, alpha in zip(p_values, alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": statistic_z, "Adjusted p-value": p_values,
                            "Adjusted alpha": alphas, "Results": results_h0})

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, all_vs_all)
    return results, figure


def finner(ranks: dict, alpha: float = 0.05, control: str = None):
    algorithm_names = list(ranks.keys())
    if not(control is None) and control not in algorithm_names:
        print(f"Warning: Control algorithm don't found, we continue All VS All")
        control = None

    all_vs_all, index_control = (True, 0) if control is None else (False, algorithm_names.index(control))
    
    results_comp = []

    if all_vs_all:
        for i in range(len(algorithm_names)):
            for j in range(i+1, len(algorithm_names)):
                comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
                # statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]])
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))
    else:
        for i in range(len(algorithm_names)):
            if index_control != i:
                comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
                # statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
                statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]])
                p_value = stats.get_p_value_normal(statistic_z)
                results_comp.append((comparisons, statistic_z, p_value))

    comparisons, statistic_z, p_values = zip(*results_comp)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])
    # TODO REVISAR EL COMO REALIZAR EL AJUSTE DE LOS ALPHAS EN VEZ DE AJUSTAR LOS P-VALUES
    alphas = [alpha] * len(p_values)
    num_comparisons = len(comparisons)
    adj_p_values = [(min(1, max([1 - (1 - p_values_with_index[j][1])**(num_comparisons/float(j+1)) for j in range(i+1)])), p_values_with_index[i][0]) for i in range(num_comparisons)]
    adj_p_values = sorted(adj_p_values, key=lambda x:x[1])
    adj_p_values, _ = zip(*adj_p_values)

    results_h0 = ["H0 is accepted" if p_value > alpha else "H0 is rejected" for p_value, alpha in zip(adj_p_values,
                                                                                                      alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": statistic_z, "p-value": p_values,
                            "Adjusted p-value": adj_p_values, "Adjusted alpha": alphas, "Results": results_h0})

    if control is None:
        control = algorithm_names[0]

    figure = generate_graph_p_values(results, control, all_vs_all)
    return results, figure


def li(ranks: dict, alpha: float = 0.05, control: str = None):
    algorithm_names = list(ranks.keys())
    if control is None or control not in algorithm_names:
        print(f"Warning: Control algorithm don't found, we continue with best ranks")
        control = algorithm_names[0]

    index_control = 0 if control is None else algorithm_names.index(control)
    # num_algorithm = len(algorithms_names)
    results_comp = []
    for i in range(len(algorithm_names)):
        if index_control != i:
            comparisons = algorithm_names[index_control] + " vs " + algorithm_names[i]
            # statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
            statistic_z = (ranks[algorithm_names[index_control]] - ranks[algorithm_names[i]])
            p_value = stats.get_p_value_normal(statistic_z)
            results_comp.append((comparisons, statistic_z, p_value))

    comparisons, statistic_z, p_values = zip(*results_comp)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_p_values = [(p_values_with_index[i][1] / (p_values_with_index[i][1] + 1 - p_values_with_index[-1][1]), p_values_with_index[i][0]) for i in range(len(p_values))]
    adj_p_values = sorted(adj_p_values, key=lambda x:x[1])
    adj_p_values, _ = zip(*adj_p_values)
    
    alphas = [alpha] * len(p_values)

    results_h0 = ["H0 is accepted" if p_value > alpha else "H0 is rejected" for p_value, alpha in zip(adj_p_values, alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": statistic_z, "p-value": p_values,
                           "Adjusted p-value": adj_p_values, "Adjusted alpha": alphas, "Results": results_h0})

    figure = generate_graph_p_values(results, control, False)
    return results, figure


def shaffer(ranks: dict, alpha: float = 0.05):
    """
    Perform a Shaffer post-hoc test using the pivot quantities obtained by a ranking test.

    Parameters
    ----------
    ranks : dict
        A dictionary with format 'groupname': rank value.
    alpha : float
    Returns
    ----------
    comparisons : list
        Strings identifier of each comparison with format 'group_i vs group_j'.
    z_values : list
        The computed Z-value statistic for each comparison.
    p_values : list
        The associated p-value from the Z-distribution.
    adjusted_p_values : list
        The associated adjusted p-values compared with a significance level.
    """

    def _calculate_independent_tests(num: int):
        """
        Calculate the number of independent test hypotheses when using All vs All strategy
        for comparing the given number of groups.
        """
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
    
    results_comp = []
    for i in range(len(algorithm_names)):
        for j in range(i+1, len(algorithm_names)):
            comparisons = algorithm_names[i] + " vs " + algorithm_names[j]
            # statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]]) / (math.sqrt((num_algorithm * (num_algorithm + 1)) / (6 * num_cases)))
            statistic_z = (ranks[algorithm_names[i]] - ranks[algorithm_names[j]])
            p_value = stats.get_p_value_normal(statistic_z)
            results_comp.append((comparisons, statistic_z, p_value))

    comparisons, statistic_z, p_values = zip(*results_comp)

    p_values_with_index = list(enumerate(p_values))
    p_values_with_index = sorted(p_values_with_index, key=lambda x: x[1])

    adj_p_values = [(min(max(t[j] * p_values_with_index[j][1] for j in range(i + 1)), 1), p_values_with_index[0]) for i in range(m)]
    adj_p_values = sorted(adj_p_values, key=lambda x: x[1])
    adj_p_values, _ = zip(*adj_p_values)

    alphas = [alpha] * len(p_values)

    results_h0 = ["H0 is accepted" if p_value > alpha else "H0 is rejected" for p_value, alpha in zip(adj_p_values,
                                                                                                      alphas)]

    results = pd.DataFrame({"Comparison": comparisons, "Statistic (Z)": statistic_z, "p-value": p_values,
                           "Adjusted p-value": adj_p_values, "Adjusted alpha": alphas, "Results": results_h0})

    control = algorithm_names[0]

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

    print(f"McNemar statistic: {mcnemar_statistic}, CV McNemar with alpha {alpha}: {cv_mcnemar}")
    if cv_mcnemar > mcnemar_statistic:
        print(f"Different distributions (reject H0) with alpha {alpha}")
    else:
        print(f"Same distributions (fail to reject H0) with alpha {alpha}")


def multiple_sign_test(dataset: pd.DataFrame, alpha: float = 0.05, criterion: bool = False, verbose: bool = False):
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

    sign_to_count = "-" if criterion else "+"
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


def contrast_estimation_based_on_medians(dataset: pd.DataFrame, criterion: bool = False, verbose: bool = False):
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

    if criterion:
        constrast_estimation_results *= -1

    if verbose:
        print(constrast_estimation_results)

    return constrast_estimation_results
