# Statistical Tests for Data Science (StaTDS)

StaTDS is a library for mathematicians, scientists, and engineers. It includes various tools to facilitate statistical analysis given a set of data samples. Within this library, you will find a wide range of statistical tests to streamline the process when conducting comparative or sample studies.

Currently, the available statistical tests are:

### **Normality**

| Name                        | Function                              |
|-----------------------------|---------------------------------------|
| Shapiro-Wilk                | parametrics.shapiro_wilk_normality    |
| D'Agostino-Pearson          | parametrics.d_agostino_pearson        |
| Kolmogorov-Smirnov          | parametrics.kolmogorov_smirnov        |


### **Homoscedasticity**

| Name                        | Function                              |
|-----------------------------|---------------------------------------|
| Levene                      | parametrics.levene                    |
| Bartlett                    | parametrics.bartlett                  |


### **Parametrics**

| Name                        | Function                              | Type Comparisons |
|-----------------------------|---------------------------------------|------------------|
| T Test paired               | parametrics.t_test_paired             | Paired           |
| T Test unpaired             | parametrics.t_test_unpaired           | Paired           |
| ANOVA between cases         | parametrics.anova_cases               | Multiple         |
| ANOVA within cases          | parametrics.anova_within_cases        | Multiple         |

### **Non Parametrics**

| Name                        | Function                              | Type Comparisons |
|-----------------------------|---------------------------------------|------------------|
| Wilcoxon                    | no_parametrics.wilconxon              | Paired           |
| Binomial Sign               | no_parametrics.binomial               | Paired           |
| Mann-Whitney U              | no_parametrics.binomial               | Paired           |
| Friedman                    | no_parametrics.friedman               | Multiple         |
| Friedman Aligned Ranks      | no_parametrics.friedman_aligned_ranks | Multiple         |
| Quade                       | no_parametrics.quade                  | Multiple         |


#### **Post-hoc**

| Name                        | Function                              |
|-----------------------------|---------------------------------------|
| Nemenyi                     | no_parametrics.nemenyi                |
| Bonferroni                  | no_parametrics.bonferroni             |
| Li                          | no_parametrics.li                     |
| Holm                        | no_parametrics.holm                   |
| Holland                     | no_parametrics.holland                |
| Finner                      | no_parametrics.finner                 |
| Hochberg                    | no_parametrics.hochberg               |
| Hommel                      | no_parametrics.hommel                 |
| Rom                         | no_parametrics.rom                    |
| Schaffer                    | no_parametrics.shaffer                |


## Authors

- [@ChrisLe7](https://www.github.com/ChrisLe7)
- [@anmoya2](https://github.com/anmoya2)
- [@jmluna](https://github.com/jmluna)
- [@sebastianventura](https://github.com/sebastianventura)


## Documentación
You can find all documentation in [Documentation Folder](https://github.com/kdis-lab/StaTDS) or [Web Docs](https://github.com/kdis-lab/StaTDS).


## Installation

StaTDS could be downloaded using two different ways: using pip or git as command line or directly from the webpage. 

### Using Git repository
The installation process for Git is detailed for each supported operating system in [1]. Additionally, a comprehensive guide on downloading StaTDS is provided. Git can be easily installed on widely used operating systems such as Windows, Mac, and Linux. It is worth noting that Git comes pre-installed on the majority of Mac and Linux machines by default.
```
 $ git clone https//github.com/kdislab/StaTDS 
```

```
    $ cd StaTDS
    $ python -m pip install --upgrade pip # To update pip
    $ python -m pip install --upgrade build # To update build
    $ python -m build 
    $ pip install dist/statds-1.0-py3-none-any.whl
```
### Using pip

Ensure that Python and pip are correctly installed on your operating system before proceeding. Once you have completed this step, utilize the following commands for library installation according to your preferred configuration:

- If you only want to use the statistical tests:
    ```shell
    $ pip install statds
    ```
- If you also want to generate PDFs:
    ```shell
    $ pip install statds[pdf]
    ```
- If you want all the features:
    ```shell
    $ pip install statds[full-app]
    ```

## References
[1] 1.5 getting started - installing git. Git. (n.d.). https://git-scm.com/book/en/v2/Getting-Started-Installing-Git 