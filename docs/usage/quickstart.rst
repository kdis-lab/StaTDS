Quickstart
==========

Installation
------------

StaTDS is compatible with Python>=3.8. We recommend installation
on a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_
environment or a `virtual env <https://docs.python.org/3/library/venv.html>`_.


Installing using conda
++++++++++++++++++++++

.. tip::

    Using conda allows to automatically solve all dependencies,
    choosing the version available supported by the system.

To install StaTDS using conda, clone the repository, navigate to the library root
directory and create a new conda environment using the provided conda configuration:

.. code:: bash

    git clone https://github.com/kdis-lab/statistical_lib.git
    cd statds
    conda env create -f conda_env.yml

Then, activate the environment and install StaTDS using :code:`pip`.

.. code:: bash

    conda activate statds
    python setup.py install  # Or 'pip install .'

.. note::

   Installation of StaTDS directly from conda is on the roadmap!


Installing using pip
++++++++++++++++++++

Alternatively, you can install the library directly from :code:`pip`. Please
refer to `PyTorch <https://pytorch.org/>`_ and `PyG installation guidelines <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_
for installation without conda. After having installed the libraries, install
:code:`statds-code` using pip. For the latest version:

.. code-block:: bash

    git clone https//github.com/kdislab/StaTDS
    python setup.py bdist_wheel
    pip install /path/to/wheelfile.whl

For the stable version:

.. code-block:: bash

    pip install statds


Example scripts
---------------

The github repository hosts `example scripts <https://github.com/kdislab/statds/tree/main/examples>`_ and `notebooks <https://github.com/kdislab/statds/tree/main/examples/notebooks>`_ on how to use the library for different use cases, such as parametrics and non-parametrics test.

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
