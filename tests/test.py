import unittest
import pandas as pd
from scipy import stats as st

import normality
import homoscedasticity
import parametrics
import no_parametrics


class Normality(unittest.TestCase):
    data = pd.read_csv("src/assets/app/sample_dataset.csv")

    def test_shapiro(self):
        for key in self.data.columns[1:]:
            sample = self.data[key].to_numpy()
            lib_statistical, lib_p_value = st.shapiro(sample)
            statistics, p_value, cv_value, hypothesis = normality.shapiro_wilk_normality(sample, alpha=0.05)
            self.assertAlmostEqual(statistics, lib_statistical, delta=0.01)
            print(statistics, lib_statistical)
            print(p_value, lib_p_value)
            self.assertAlmostEqual(p_value, lib_p_value, delta=0.1)

    def test_skewness(self):
        for key in self.data.columns[1:]:
            sample = self.data[key].to_numpy()
            lib_skew = st.skew(sample)
            skew = normality.skewness(sample)
            self.assertAlmostEqual(skew, lib_skew, delta=0.0001)
            lib_z_skew, lib_p_value = st.skewtest(sample)
            z_skew, p_value = normality.skewness_test(skew, len(sample))
            self.assertAlmostEqual(z_skew, lib_z_skew, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_value, delta=0.0001)

    def test_kurtosis(self):
        for key in self.data.columns[1:]:
            sample = self.data[key].to_numpy()
            lib_kurtosis = st.kurtosis(sample, fisher=False)
            kurtosis = normality.kurtosis(sample)
            self.assertAlmostEqual(kurtosis, lib_kurtosis, delta=0.0001)
            lib_z_skew, lib_p_value = st.kurtosistest(sample)
            z_kurtosis, p_value = normality.kurtosis_test(kurtosis, len(sample))
            self.assertAlmostEqual(z_kurtosis, lib_z_skew, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_value, delta=0.0001)

    def test_d_agostino_pearson(self):
        for key in self.data.columns[1:]:
            sample = self.data[key].to_numpy()
            lib_statistic, lib_p_valor = st.normaltest(sample)
            statistic, p_value, cv_value, hypothesis = normality.d_agostino_pearson(sample)
            self.assertAlmostEqual(statistic, lib_statistic, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_valor, delta=0.0001)

    def test_kolmogor_smirnow(self):
        for key in self.data.columns[1:]:
            sample = self.data[key].to_numpy()
            lib_statistic, lib_p_valor = st.kstest(sample, 'norm')
            statistic, p_value, cv_value, hypothesis = normality.kolmogorov_smirnov(sample)
            # statistic, p_value, cv_value, hypothesis = parametrics.kolmogorov_smirnov_2(sample)
            self.assertAlmostEqual(statistic, lib_statistic, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_valor, delta=0.0001)


class Homoscedasticity(unittest.TestCase):
    data = pd.read_csv("src/assets/app/sample_dataset.csv")

    def test_levene(self):
        samples = [self.data[i].to_numpy() for i in self.data.columns[1:]]
        lib_statistic, lib_p_valor = st.levene(*samples, center="mean")
        statistic, p_value, cv_value, hypothesis = homoscedasticity.levene_test(self.data, center="mean")
        self.assertAlmostEqual(statistic, lib_statistic, delta=0.00001)
        self.assertAlmostEqual(statistic, lib_statistic, delta=0.00001)
        self.assertAlmostEqual(p_value, lib_p_valor, delta=0.00001)

    def test_bartlett(self):
        samples = [self.data[i].to_numpy() for i in self.data.columns[1:]]
        lib_statistic, lib_p_valor = st.bartlett(*samples)
        statistic, p_value, cv_value, hypothesis = homoscedasticity.bartlett_test(self.data)
        self.assertAlmostEqual(statistic, lib_statistic, delta=0.0001)
        self.assertAlmostEqual(p_value, lib_p_valor, delta=0.0001)


class Parametrics(unittest.TestCase):
    data = pd.read_csv("src/assets/app/sample_dataset.csv")

    def test_t_test_paired(self):
        columns = list(self.data.columns[1:])
        pairwise_keys = [[columns[i], columns[j]] for i in range(len(columns)) for j in range(i+1, len(columns))]
        for keys in pairwise_keys:
            sample = self.data[keys]
            samples = [sample[i].to_numpy() for i in sample.columns]
            lib_statistic, lib_p_valor = st.ttest_rel(*samples)
            statistic, cv_value, p_value, hypothesis = parametrics.t_test_paired(sample)
            self.assertAlmostEqual(statistic, lib_statistic, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_valor, delta=0.0001)

    def test_t_test_unpaired(self):
        columns = list(self.data.columns[1:])
        pairwise_keys = [[columns[i], columns[j]] for i in range(len(columns)) for j in range(i+1, len(columns))]
        for keys in pairwise_keys:
            sample = self.data[keys]
            samples = [sample[i].to_numpy() for i in sample.columns]
            lib_statistic, lib_p_valor = st.ttest_ind(*samples)
            statistic, cv_value, p_value, hypothesis = parametrics.t_test_unpaired(sample)
            self.assertAlmostEqual(statistic, lib_statistic, delta=0.0001)
            self.assertAlmostEqual(p_value, lib_p_valor, delta=0.0001)

    # TODO Falta comparar con la librería de scipy, hay que buscar cual es la más adecuada
    def test_anova(self):
        tables_summary, statistic, p_value, cv_value, hypothesis = parametrics.anova_cases(self.data)
        self.assertEqual(len(tables_summary), 2)
        self.assertTrue(isinstance(tables_summary[0], pd.DataFrame))
        self.assertTrue(isinstance(tables_summary[1], pd.DataFrame))

    def test_anova_within_cases(self):
        tables_summary, statistic, p_value, cv_value, hypothesis = parametrics.anova_within_cases(self.data)
        self.assertEqual(len(tables_summary), 2)
        self.assertTrue(isinstance(tables_summary[0], pd.DataFrame))
        self.assertTrue(isinstance(tables_summary[1], pd.DataFrame))


class NoParametrics(unittest.TestCase):
    data = pd.read_csv("src/assets/app/sample_dataset.csv")

    def test_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        results_article = {"ranks": {"PDFC": 1.7708333333333337, "NNEP": 2.479166666666667,
                                     "IS-CHC + 1NN": 2.479166666666667, "FH-GBML": 3.2708333333333326},
                           "statistic": 16.225, "p_value": 0.001019673101913754, "iman_p_value": 4.970002674994732E-4}

        for key in ranks.keys():
            self.assertAlmostEqual(ranks[key], results_article["ranks"][key], delta=0.0001)
        self.assertAlmostEqual(stadistic, results_article["statistic"], delta=0.0001)
        self.assertAlmostEqual(p_value, results_article["p_value"], delta=0.0001)

    def test_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        results_article = {"ranks": {"PDFC": 29.333333333333332, "NNEP": 46.79166666666666,
                                     "IS-CHC + 1NN": 46.97916666666667, "FH-GBML": 70.89583333333334},
                           "statistic": 18.84087854240361,
                           "p_value": 2.949111303799379E-4}
        for key in ranks.keys():
            self.assertAlmostEqual(ranks[key], results_article["ranks"][key], delta=0.000001)
        self.assertAlmostEqual(stadistic, results_article["statistic"], delta=0.0001)
        self.assertAlmostEqual(p_value, results_article["p_value"], delta=0.0001)

    def test_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05, criterion=True)
        results_article = {"ranks": {"PDFC": 1.3883333333333336, "NNEP": 2.5383333333333336,
                                     "IS-CHC + 1NN": 2.591666666666667, "FH-GBML": 3.481666666666666},
                           "statistic": 22.206729531787442,
                           "p_value": 3.5726369523928085E-10}

        for key in ranks.keys():
            self.assertAlmostEqual(ranks[key], results_article["ranks"][key], delta=0.000001)
        self.assertAlmostEqual(stadistic, results_article["statistic"], delta=0.0001)
        self.assertAlmostEqual(p_value, results_article["p_value"], delta=0.0001)

    def test_bonferroni_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.bonferroni(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                                type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70982e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.17204},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.17204}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holm_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.holm(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                          type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70982e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.11469},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.11469}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hochberg_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.hochberg(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                              type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.709823486999458E-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.057346851901366444},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.057346851901366444}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hommel_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.hommel(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70982e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.05735},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.05735}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holland_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.holland(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                             type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70973e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.11141},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.11141}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_rom_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.rom(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                         type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70982e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.05735},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.05735}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_finner_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.finner(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.69941e-5, "Adjusted p-value": 1.70982e-4},
                           "IS-CHC + 1NN": {"p_value": 0.05735, "Adjusted p-value": 0.08477},
                           "NNEP": {"p_value": 0.05735, "Adjusted p-value": 0.08477}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_li_friedman(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman(self.data, 0.05,
                                                                                  criterion=True)
        result, fig = no_parametrics.li(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                        type_rank="Friedman")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 5.699411623331527E-5, "Adjusted p-value": 6.045773104701031E-5},
                           "IS-CHC + 1NN": {"p_value": 0.057346851901366444, "Adjusted p-value": 0.057346851901366444},
                           "NNEP": {"p_value": 0.057346851901366444, "Adjusted p-value": 0.057346851901366444}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_bonferroni_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.bonferroni(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                                type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080804003379604E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.08463508846642964},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.08979043348674859}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holm_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.holm(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                          type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080804003379604E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.0564233923109531},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.0564233923109531}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hochberg_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.hochberg(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                              type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080804003379604E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.029930144495582865},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.029930144495582865}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hommel_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)

        result, fig = no_parametrics.hommel(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080804003379604E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.029930144495582865},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.029930144495582865}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holland_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.holland(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                             type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080802332248837E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.0556274925109842},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.0556274925109842}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_rom_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.rom(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                         type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080804003379604E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.029930144495582865},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.029930144495582865}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_finner_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.finner(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 7.080802332248837E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.04201766339344748},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.04201766339344748}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_li_friedman_aligned_ranks(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.friedman_aligned_ranks(self.data, 0.05,
                                                                                                criterion=True)
        result, fig = no_parametrics.li(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                        type_rank="Friedman Aligned Ranks")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 2.3602680011265347E-7, "Adjusted p-value": 2.4330901671248775E-7},
                           "IS-CHC + 1NN": {"p_value": 0.02821169615547655, "Adjusted p-value": 0.02826025995228252},
                           "NNEP": {"p_value": 0.029930144495582865, "Adjusted p-value": 0.029930144495582865}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_bonferroni_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.bonferroni(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                                type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.8050869640761657E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.0632741882686326},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.0825468427075807}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holm_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.holm(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                          type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.8050869640761657E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.0421827921790884},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.0421827921790884}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hochberg_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)

        result, fig = no_parametrics.hochberg(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                              type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.8050869640761657E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.02751561423586023},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.02751561423586023}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_hommel_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)

        result, fig = no_parametrics.hommel(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.8050869640761657E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.02751561423586023},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.02751561423586023}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_holland_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.holland(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                             type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.804978354955633E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.04173794519008245},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.04173794519008245}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_rom_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.rom(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                         type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.8050869640761657E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.02751561423586023},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.02751561423586023}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_finner_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.finner(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                            type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 1.804978354955633E-4},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.03146968542314532},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.03146968542314532}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)

    def test_li_quade(self):
        ranks, stadistic, p_value, cv_value, hypothesis = no_parametrics.quade(self.data, 0.05,
                                                                               criterion=True)
        result, fig = no_parametrics.li(ranks, self.data.shape[0], 0.05, control=self.data.columns[1],
                                        type_rank="Quade")
        names = result["Comparison"].to_numpy()
        p_values = result["p-value"].to_numpy()
        adj_p_values = result["Adjusted p-value"].to_numpy()
        names = [i[i.find("vs ") + 3:] for i in names]
        results = {key: {"p_value": p, "Adjusted p-value": adj_p_value} for key, p, adj_p_value in
                   zip(names, p_values, adj_p_values)}

        results_article = {"FH-GBML": {"p_value": 6.0169565469205524E-5, "Adjusted p-value": 6.186818397108111E-5},
                           "IS-CHC + 1NN": {"p_value": 0.0210913960895442, "Adjusted p-value": 0.021227767901301525},
                           "NNEP": {"p_value": 0.02751561423586023, "Adjusted p-value": 0.027515614235860235}}

        for key in results.keys():
            self.assertAlmostEqual(results[key]["p_value"], results_article[key]["p_value"], delta=0.0001)
            self.assertAlmostEqual(results[key]["Adjusted p-value"], results_article[key]["Adjusted p-value"],
                                   delta=0.0001)


if __name__ == '__main__':
    unittest.main()
