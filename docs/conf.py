# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import sys
import os
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
sys.path.insert(0, os.path.abspath('../'))

project = 'StaTDS'
copyright = '2023, Christian Luna Escudero, Antonio Rafael Moya Martín-Castaño, José María Luna Ariza, Sebastián Ventura Soto'
author = 'Christian Luna Escudero, Antonio Rafael Moya Martín-Castaño, José María Luna Ariza, Sebastián Ventura Soto'
release = 'GPLv3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration', 'sphinx.ext.doctest', 'sphinx.ext.autodoc',]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_baseurl = ''
html_logo = '_static/img/logo-StaTDS-without-background.png'
html_favicon = '_static/img/logo-StaTDS.png'

html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2e5c7d",
        "color-brand-content": "#2e5c7d",
        "codebgcolor": "red",
        "codetextcolor": "red",
    },
    "dark_css_variables": {
        "color-brand-primary": "#6998b4",
        "color-brand-content": "#6998b4",
        "codebgcolor": "green",
        "codetextcolor": "green",
    }

}