from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='cpdetect',
    version='0.0.1',
    description='A package containing multiple change-point detection methods for normal mean model (mean shift detection).',
    long_description=long_description,
    author='Szymon Malec',
    author_email='szymon.malec@o2.pl',
    url='"https://github.com/Szymex49/cpdetect"',
    license='GPLv3',
    packages=find_packages(),
    package_data={'': ['binseg/quantiles/Z.csv', 'binseg/quantiles/T.csv', 'sara/quantiles/Z.csv', 'sara/quantiles/T.csv']},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'pandas'],
    python_requires='>=3.8',
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ]
)
