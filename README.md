# ![https://github.com/kdis-lab/StaTDS](https://raw.githubusercontent.com/kdis-lab/StaTDS/master/assets/statds.png) Hi, StaTDS is a library for statistical testing and comparison of algorithm results 👋
## Statistical Tests for Data Science (StaTDS)

StaTDS is a library for mathematicians, scientists, and engineers. It includes various tools to facilitate statistical analysis given a set of data samples. Within this library, you will find a wide range of statistical tests to streamline the process when conducting comparative or sample studies.

![https://github.com/kdis-lab/StaTDS](https://raw.githubusercontent.com/kdis-lab/StaTDS/master/assets/banner-lib.png)

[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UCRuhSEuWAKQuCLQmoFim5Hw)](https://www.youtube.com/@StaTDS?sub_confirmation=1) ![GitHub Followers](https://img.shields.io/github/followers/kdis-lab?style=social) ![GitHub Followers](https://img.shields.io/github/stars/kdis-lab/StaTDS?style=social)  ![Pypi download](https://img.shields.io/pypi/dm/statds)

### Available statistical test

#### **Normality**

| Name                        | Function                              |
|-----------------------------|---------------------------------------|
| Shapiro-Wilk                | normality.shapiro_wilk_normality      |
| D'Agostino-Pearson          | normality.d_agostino_pearson          |
| Kolmogorov-Smirnov          | normality.kolmogorov_smirnov          |


#### **Homoscedasticity**

| Name                        | Function                              |
|-----------------------------|---------------------------------------|
| Levene                      | homoscedasticity.levene               |
| Bartlett                    | homoscedasticity.bartlett             |


#### **Parametrics**

| Name                        | Function                              | Type Comparisons |
|-----------------------------|---------------------------------------|------------------|
| T Test paired               | parametrics.t_test_paired             | Paired           |
| T Test unpaired             | parametrics.t_test_unpaired           | Paired           |
| ANOVA between cases         | parametrics.anova_cases               | Multiple         |
| ANOVA within cases          | parametrics.anova_within_cases        | Multiple         |

#### **Non Parametrics**

| Name                        | Function                              | Type Comparisons |
|-----------------------------|---------------------------------------|------------------|
| Wilcoxon                    | no_parametrics.wilconxon              | Paired           |
| Binomial Sign               | no_parametrics.binomial               | Paired           |
| Mann-Whitney U              | no_parametrics.mannwhitneyu           | Paired           |
| Friedman                    | no_parametrics.friedman               | Multiple         |
| Friedman + Iman-Davenport   | no_parametrics.iman_davenport         | Multiple         |
| Friedman Aligned Ranks      | no_parametrics.friedman_aligned_ranks | Multiple         |
| Quade                       | no_parametrics.quade                  | Multiple         |
| Kruskal-Wallis              | no_parametrics.kruskal_wallis         | Multiple         |


##### **Post-hoc**

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
| Shaffer                     | no_parametrics.shaffer                |

## Developed in:
![Python](https://img.shields.io/badge/Python-yellow?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)

## Authors

- [@ChrisLe7](https://www.github.com/ChrisLe7)
- [@anmoya2](https://github.com/anmoya2)
- [@jmluna](https://github.com/jmluna)
- [@sebastianventura](https://github.com/sebastianventura)


## Documentación
You can find all documentation in [Documentation Folder](https://github.com/kdis-lab/StaTDS), [Web Docs](https://github.com/kdis-lab/StaTDS) or [Youtube Channel](https://www.youtube.com/@StaTDS).


## Installation

StaTDS could be downloaded using two different ways: using pip or git as command line or docker container. 

### Using Git repository
The installation process for Git is detailed for each supported operating system in [1]. Additionally, a comprehensive guide on downloading StaTDS is provided. Git can be easily installed on widely used operating systems such as Windows, Mac, and Linux. It is worth noting that Git comes pre-installed on the majority of Mac and Linux machines by default.
```
 $ git clone https//github.com/kdis-lab/StaTDS 
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

## Quick start

- If you have questions, please ask them in GitHub Discussions.
- If you want to report a bug, please open an [issue on the GitHub repository](https://github.com/kdis-lab/StaTDS/issues).
- If you want to see StaTDS in action, please click on the link below and navigate to the notebooks/ folder to open a collection of interactive Jupyter notebooks.

### Using StaTDS Library - API

#### Normality tests: Shapiro Test

```python
from statds.normality import shapiro_wilk_normality
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
results = []

for i in range(1, len(columns)): 
    results.append(shapiro_wilk_normality(dataset[columns[i]].to_numpy(), alpha))

statistic_list, p_value_list, cv_value_list, hypothesis_list = zip(*results)

results_test = pd.DataFrame({"Algorithm": columns[1:], "Statistic": statistic_list, "p-value": p_value_list, "Results": hypothesis_list})
print(results_test)
```

#### Homoscedasticy tests: Levene
```python
from statds.homoscedasticity import levene_test
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
statistic, p_value, rejected_value, hypothesis = levene_test(dataset, alpha, center='mean')
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}")
```


#### Parametrics tests: T-test
```python
from statds.parametrics import t_test_paired
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
selected_columns = [columns[1], columns[2]]
statistic, rejected_value, p_value, hypothesis = t_test_paired(dataset[selected_columns], alpha)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
```

#### Parametrics tests: ANOVA
```python
from statds.parametrics import anova_test
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
statistic, p_value, rejected_value, hypothesis = anova_test(dataset, alpha)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
```

#### Non-parametrics tests: Wilcoxon

```python
from statds.no_parametrics import wilcoxon

dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
selected_columns = [columns[1], columns[2]]
statistic, p_value, rejected_value, hypothesis = wilcoxon(dataset[selected_columns], alpha)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
```

#### Non-parametrics tests: Friedman Test

```python
import pandas as pd
from statds.no_parametrics import friedman

dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, minimize=False)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
print(rankings)
```

#### Post-hoc tests: Bonferroni

```python
from statds.no_parametrics import friedman, bonferroni
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, minimize=False)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
print(rankings)
num_cases = dataset.shape[0]
results, figure = bonferroni(rankings, num_cases, alpha, control = None, type_rank = "Friedman")
print(results)
figure.show()
```

#### Post-hoc tests: Nemenyi

```python
from statds.no_parametrics import friedman, nemenyi
dataset = pd.read_csv("dataset.csv")
alpha = 0.05
columns = list(dataset.columns)
rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, minimize=False)
print(hypothesis)
print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
print(rankings)
num_cases = dataset.shape[0]
ranks_values, critical_distance_nemenyi, figure = nemenyi(rankings, num_cases, alpha)
print(ranks_values)
print(critical_distance_nemenyi)
figure.show()
```


### Using StaTDS Web Client

#### Local with Python

You only need create a python script with next code:

```python
from statds import app

app.start_app(port=8050)
```

Now, you can access to the interface with your Web navigator through the following url: http://localhost:8050

#### Local Using Docker

Firstly, to begin with, it is essential to download the repository from GitHub to obtain the Dockerfile. Before this step, ensure that Docker is installed on your computer [2]. With Docker ready to use, you can build the application's image by executing the following command:

```shell
docker build -t name-lib ./
```

After the image has been successfully created, the next step is to instantiate a container using that image.
 
```shell
docker run -p 8050:8050 --name container name-lib
```

Now, you can access to the interface with your Web navigator through the following url: http://localhost:8050


## References
[1] 1.5 getting started - installing git. Git. (n.d.). https://git-scm.com/book/en/v2/Getting-Started-Installing-Git 
[2] Get Docker — Docker Docs. Docker Inc. 2023. url: https://docs.docker.com/get-docker