# statistical_lib

statistical_lib is a library for mathematicians, scientists, and engineers. It includes various tools to facilitate statistical analysis given a set of data samples. Within this library, you will find a wide range of statistical tests to streamline the process when conducting comparative or sample studies.

Currently, the available statistical tests are:

|                         |                             |                                       |                  |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| Type Test               | Name                        | Name                                  | Type Comparisons |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| **Normality**           |                             |                                       |                  |
|                         | Shapiro-Wilk                | parametrics.shapiro_wilk_normality    |                  |
|                         | D'Agostino-Pearson          | parametrics.d_agostino_pearson        |                  |
|                         | Kolmogorov-Smirnov          | parametrics.kolmogorov_smirnov        |                  |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| **Homoscedasticity**    |                             |                                       |                  |
|                         | Levene                      | parametrics.levene                    |                  |
|                         | Bartlett                    | parametrics.bartlett                  |                  |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| **Parametrics**         | T Test paired               | parametrics.t_test_paired             | Paired           |
|                         | T Test unpaired             | parametrics.t_test_unpaired           | Paired           |
|                         | ANOVA between cases         | parametrics.anova_cases               | Multiple         |
|                         | ANOVA within cases          | parametrics.anova_within_cases        | Multiple         |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| **Non Parametrics**     |                             |                                       |                  |
|                         | Wilcoxon                    | no_parametrics.wilconxon              | Paired           |
|                         | Binomial Sign               | no_parametrics.binomial               | Paired           |
|                         | Mann-Whitney U              | no_parametrics.binomial               | Paired           |
|                         | Friedman                    | no_parametrics.friedman               | Multiple         |
|                         | Friedman Aligned Ranks      | no_parametrics.friedman_aligned_ranks | Multiple         |
|                         | Quade                       | no_parametrics.quade                  | Multiple         |
|-------------------------|-----------------------------|---------------------------------------|------------------|
| **Post Hoc**            |                             |                                       |                  |
|                         | Nemenyi                     | no_parametrics.nemenyi                |                  |
|                         | Bonferroni                  | no_parametrics.bonferroni             |                  |
|                         | Li                          | no_parametrics.li                     |                  |
|                         | Holm                        | no_parametrics.holm                   |                  |
|                         | Holland                     | no_parametrics.holland                |                  |
|                         | Finner                      | no_parametrics.finner                 |                  |
|                         | Hochberg                    | no_parametrics.hochberg               |                  |
|                         | Hommel                      | no_parametrics.hommel                 |                  |
|                         | Rom                         | no_parametrics.rom                    |                  |
|                         | Schaffer                    | no_parametrics.shaffer                |                  |
|-------------------------|-----------------------------|---------------------------------------|------------------|



## Authors

- [@ChrisLe7](https://www.github.com/ChrisLe7)
- [Antonio R. Moya Martín-Castaño](https://github.com/anmoya2)
- [@jmluna](https://github.com/jmluna)
- [@sebastianventura](https://github.com/sebastianventura)


## Documentación

[Documentation](https://github.com/kdis-lab/statistical_lib)



## Installation

statistical_lib could be downloaded using two different ways: using pip or git as command line or directly from the webpage. 

### Using Git repository

### Using pip
```shell
pip install statistical_lib
```