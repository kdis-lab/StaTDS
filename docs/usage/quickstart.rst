Quickstart
==========

Installation
------------

StaTDS is compatible with Python>=3.8. We recommend installation
on a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_
environment or a `virtual env <https://docs.python.org/3/library/venv.html>`_.


Installing using pip
++++++++++++++++++++

You can install the library directly from :code:`pip`. 

For the latest version:

.. code-block:: bash

    git clone https//github.com/kdislab/StaTDS
    python setup.py bdist_wheel
    pip install /path/to/wheelfile.whl

For the stable version:

- If you only want to use the statistical tests:

.. code-block:: bash

    pip install statds[pdf]

- If you also want to generate PDFs:

.. code-block:: bash

    pip install statds[pdf]

- If you want all the features:

.. code-block:: bash

    pip install statds[full-app]


Example scripts
---------------

The github repository hosts `example scripts <https://github.com/kdislab/statds/tree/main>`_ and `notebooks <https://github.com/kdislab/statds/tree/main/notebooks>`_ on how to use the library for different use cases, such as parametrics and non-parametrics test. Here you can see some examples:

Normality tests: Shapiro Test
+++++++++++++++++++++++++++++

.. code-block:: python

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


Homoscedasticy tests: Levene
++++++++++++++++++++++++++++

.. code-block:: python

    from statds.homoscedasticity import levene_test
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    statistic, p_value, rejected_value, hypothesis = levene_test(dataset, alpha, center='mean')
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}")



Parametrics tests: T-test
+++++++++++++++++++++++++

.. code-block:: python

    from statds.parametrics import t_test_paired
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    selected_columns = [columns[1], columns[2]]
    statistic, rejected_value, p_value, hypothesis = t_test_paired(dataset[selected_columns], alpha)
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")

Parametrics tests: ANOVA
++++++++++++++++++++++++

.. code-block:: python

    from statds.parametrics import anova_test
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    statistic, p_value, rejected_value, hypothesis = anova_test(dataset, alpha)
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")

Non-parametrics tests: Wilcoxon
+++++++++++++++++++++++++++++++

.. code-block:: python

    from statds.no_parametrics import wilconxon
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    selected_columns = [columns[1], columns[2]]
    statistic, p_value, rejected_value, hypothesis = wilconxon(dataset[selected_columns], alpha)
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")


Non-parametrics tests: Friedman Test
++++++++++++++++++++++++++++++++++++

.. code-block:: python

    import pandas as pd
    from statds.no_parametrics import friedman

    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, criterion=False)
    print(hypothesis)
    print(ff"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
    print(rankings)

Post-hoc tests: Bonferroni
++++++++++++++++++++++++++

.. code-block:: python

    from statds.no_parametrics import friedman, bonferroni
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, criterion=False)
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
    print(rankings)
    num_cases = dataset.shape[0]
    results, figure = bonferroni(rankings, num_cases, alpha, control = None, type_rank = "Friedman")
    print(results)
    figure.show()

Post-hoc tests: Nemenyi
+++++++++++++++++++++++

.. code-block:: python

    from statds.no_parametrics import friedman, nemenyi
    dataset = pd.read_csv("dataset.csv")
    alpha = 0.05
    columns = list(dataset.columns)
    rankings, statistic, p_value, critical_value, hypothesis = friedman(dataset, alpha, criterion=False)
    print(hypothesis)
    print(f"Statistic {statistic}, Rejected Value {rejected_value}, p-value {p_value}")
    print(rankings)
    num_cases = dataset.shape[0]
    ranks_values, critical_distance_nemenyi, figure = nemenyi(rankings, num_cases, alpha)
    print(ranks_values)
    print(critical_distance_nemenyi)
    figure.show()


Citing
------

If you use StaTDS for your research, please consider citing the library

Bibtex entry::

   @InProceedings{statds,
     author={Christian Luna Escudero, Antonio Rafael Moya Martín-Castaño, José María Luna Ariza, Sebastián Ventura Soto},
     title={{StaTDS}: Statistical Tests for Data Science (name article and journal)},
     booktitle={journal},
     year={2023}
   }
