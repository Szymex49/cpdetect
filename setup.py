from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='cpdetect',
    version='0.0.1',
    description='A package containing multiple change-point detection methods for normal mean model (mean shift detection).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Szymon Malec',
    author_email='szymon.malec@o2.pl',
    url="https://github.com/Szymex49/cpdetect",
    license='GPLv3',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={"cpdetect": ["binseg/quantiles/binseg_Z.csv", "binseg/quantiles/binseg_T.csv", "sara/quantiles/sara_Z.csv", "sara/quantiles/sara_T.csv"]},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'pandas'],
    python_requires='>=3.8',
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    ]
)
