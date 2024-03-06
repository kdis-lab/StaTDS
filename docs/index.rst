.. StaTDS documentation master file, created by
   sphinx-quickstart on Mon Dec  4 15:49:18 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/kdis-lab/statistical_lib/tree/dev

Statistical Tests for Data Science (StaTDS)
===========================================


StaTDS is a python library for statistical testing and comparison of algorithm results.

StaTDS stands out as a distinctive Python library geared towards the statistical comparison of algorithms, built entirely in pure Python. StaTDS guarantees its perfect functionality and stability by not depending on external libraries, a common practice in some similar libraries. It supports an extensive range of statistical tests (23 different tests) for diverse use cases, so it outperforms existing libraries for statistical tests. The library includes some tests to determine whether parametric or non-parametric tests the user should use (normality and homoscedasticity test). Not only is it meticulously crafted for Data Scientists, but StaTDS also extends its utility as a web application so a broader audience can use it, ensuring accessibility and ease of use for all users.

In detail, the package provide:

**Normality Test**:
+++++++++++++++++++
   - Shapiro-Wilk. :func:`statds.normality.shapiro_wilk_normality`
   - D'Agostino-Pearson. :func:`statds.normality.d_agostino_pearson`
   - Kolmogorov-Smirnov. :func:`statds.normality.kolmogorov_smirnov`
   
**Homoscedasticity Test**:
++++++++++++++++++++++++++
   - Levene. :func:`statds.homoscedasticity.levene_test`
   - Bartlett. :func:`statds.homoscedasticity.bartlett_test`

**Parametrics Test**:
+++++++++++++++++++++
   - T Test paired. :func:`statds.parametrics.t_test_paired`
   - T Test unpaired. :func:`statds.parametrics.t_test_unpaired`
   - ANOVA between cases. :func:`statds.parametrics.anova_cases`
   - ANOVA within cases. :func:`statds.parametrics.anova_within_cases`

**Non Parametrics Test**:
+++++++++++++++++++++++++
   - Wilcoxon. :func:`statds.no_parametrics.wilconxon`
   - Binomial Sign. :func:`statds.no_parametrics.binomial`
   - Mann-Whitney U. :func:`statds.no_parametrics.mannwhitneyu`
   - Friedman. :func:`statds.no_parametrics.friedman`
   - Friedman + Iman Davenport. :func:`statds.no_parametrics.iman_davenport`
   - Friedman Aligned Ranks. :func:`statds.no_parametrics.friedman_aligned_ranks`
   - Quade. :func:`statds.no_parametrics.quade`
   - Kruskal-Wallis. :func:`statds.no_parametrics.kruskal_wallis`

**Post-hoc**
++++++++++++
   - Nemenyi. :func:`statds.no_parametrics.nemenyi`
   - Bonferroni. :func:`statds.no_parametrics.bonferroni`
   - Li. :func:`statds.no_parametrics.li`
   - Holm. :func:`statds.no_parametrics.holm`
   - Holland. :func:`statds.no_parametrics.holland`
   - Finner. :func:`statds.no_parametrics.finner`
   - Hochberg. :func:`statds.no_parametrics.hochberg`
   - Hommel. :func:`statds.no_parametrics.hommel`
   - Rom. :func:`statds.no_parametrics.rom`
   - Schaffer. :func:`statds.no_parametrics.shaffer`

.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage
   :hidden:

   usage/quickstart
   usage/notebooks

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package API
   :hidden:

   modules

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Index
   :hidden:

   genindex
   py-modindex