Notebooks
=========
This tutorial series enhances the accessibility of \nameLibrary \ by offering a curated collection of Jupyter notebooks. You can access these notebooks on GitHub at `link <https://github.com/kdis-lab/statds>`_ or Google Colab. Each section below outlines the key topics covered in the respective notebooks.	


`StaTDS Tutorial: Classification <https://drive.google.com/file/d/1DRxkLaYEAwqTLJMNp8uyzm2tyitaQ6DR/view?usp=sharing>`_
-----------------------------------------------------------------------------------------------------------------------
Explore a side-by-side comparison of four classification algorithms: Random Forest, Gradient Boosting, Support Vector Machine, and Artificial Neural Network. Our focus is on measuring accuracy and false positive rates. We conduct a series of tests on ten diverse datasets, using 5-fold cross-validation to evaluate algorithmic performance.

`StaTDS Tutorial: Regression <https://drive.google.com/file/d/1FUc1S7P9E_L-fOGyoStXs6mXLy0FqLj8/view?usp=sharing>`_
-------------------------------------------------------------------------------------------------------------------

This section presents a comparative analysis of four regression algorithms: Random Forest, Support Vector Machine, Lasso Regression, and Ridge Regression. We evaluate these using Mean Squared Error (MSE) and R-squared (R2). We examine the algorithms' behaviors on ten datasets, employing a bootstrap evaluation method with a size of 5.


`StaTDS Tutorial: Clustering <https://drive.google.com/file/d/1MjP4vT7ar14Qcd5Q5yebdSt0yLGQOTfb/view?usp=sharing>`_
-------------------------------------------------------------------------------------------------------------------

In this tutorial, we compare three clustering algorithms: K-Means, Agglomerative Clustering, and DBSCAN, with an emphasis on Silhouette Score and Davies-Bouldin Score as evaluation metrics. We implement various tests to assess the performance of these algorithms over ten datasets.


`StaTDS Tutorial: Association Rule Mining <https://drive.google.com/file/d/17LgUCNTCsxLCK4X3cwzytXG9WjPOURgX/view?usp=sharing>`_
--------------------------------------------------------------------------------------------------------------------------------

This Jupyter notebook presents a comparison of two well-known algorithms for mining association rules: Apriori and FP-Growth. The algorithms are compared in terms of runtime (the lower the better) for a varied set of datasets. The notebook includes some specific parts implemented by pandas (dataset loading) and mlxtend (association rule mining algorithms) that do not belong to StaTDS and could be done with other different libraries.


`StaTDS Tutorial: Preprocessing methods <https://colab.research.google.com/drive/18QhflKEKTBaJHCeT4g-MZyfL2-1CFiZs?usp=sharing>`_
---------------------------------------------------------------------------------------------------------------------------------

This Jupyter notebook provides a comparative analysis of various data preprocessing techniques such as normalization, standardization, binarization, and PCA. The performance of these methods is evaluated based on their impact on the accuracy and efficiency of a set of machine learning algorithms across multiple datasets. The notebook includes some specific parts implemented by pandas (dataset loading) and sklearn (preprocessing algorithms) that do not belong to StaTDS and could be done with other different libraries.


`StaTDS Tutorial: Feature selection methods <https://colab.research.google.com/drive/18QhflKEKTBaJHCeT4g-MZyfL2-1CFiZs?usp=sharing>`_
-------------------------------------------------------------------------------------------------------------------------------------

In this Jupyter notebook, we examine different feature selection techniques including SelectKBest, RFE, and SelectFromModel. These methods are evaluated to determine their effectiveness in improving the performance of a Random Forest classifier across various datasets. The evaluation metrics focus on model accuracy and computational cost. The notebook includes some specific parts implemented by pandas (dataset loading) and sklearn (preprocessing, decomposition algorithms) that do not belong to StaTDS and could be done with other different libraries.