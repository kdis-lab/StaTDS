"""
Created on Tu Sep 19
Python X Setup.
"""

from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    "pandas>=2.0",
    "numpy>=1.20",
    "matplotlib>=3.7"
]

extras_require = {
    'generate_pdf': [
        'fpdf2==2.7.5',  
    ],
    'full-app': [
        'dash==2.13.0',
        'dash_bootstrap_components==1.4.2',
        'dash_daq==0.5.0',
        'dash_ag_grid==2.3.0',
        'fpdf2==2.7.5'
    ]
}

setup(
    #  Project name.
    #  $ pip install statds
    name='statds',

    # Version
    version='1.0',

    # Description
    description='Library for statistical testing and comparison of algorithm results',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/kdis-lab/StaTDS',

    # Author
    author='Christian Luna Escudero, Antonio R. Moya Martín-Castaño, José María Luna Ariza, Sebastián Ventura Soto',

    # Author email
    author_email='i82luesc@uco.es, sventura@uco.es',

    # Keywords
    keywords=['Test Statistical',
              'Parametrics Tests',
              'No Parametrics Tests',
              'Comparare Algorithms Results',
              'Post-hoc Tests'],

    # Packages # Excluimos las carpetas del código que no se usen en la librería
    package_dir={"": "./"}, # CUIDADO CON ESTE DIRECTORIO, DEBE DE SER EL DE ANTES
    packages=find_packages(where="./", exclude=['app', 'docs', 'tests', 'examples', 'src']),
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.rst', '*.csv', '*.png', '*.css'],  # tipos de archivos a incluir en todos los paquetes
        'statds': ['assets/**/*'],  # incluye todo en la carpeta assets
    },
    # Test suite
    test_suite='test',

    # Requeriments
    install_requires=install_requires,
    extras_require=extras_require,
    long_description_content_type='text/markdown'
)
