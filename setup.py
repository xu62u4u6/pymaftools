from setuptools import setup, find_packages

setup(
    name='pymaftools',
    version='0.2.3',
    author = "xu62u4u6",
    python_requires='>=3.8',
    author_email="199928ltyos@gmail.com",
    description='pymaftools is a Python package for handling and analyzing Mutation Annotation Format (MAF) files. It provides utilities for data manipulation and visualization, including classes for MAF parsing, oncoplot generation, and additional plots like lollipop and boxplots with statistical testing.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=["pymaftools", "pymaftools.*"]),
    include_package_data=True,
    install_requires=[
        'pandas>2.0', 
        'numpy', 
        'networkx', 
        'matplotlib', 
        'seaborn', 
        'statannotations',
        'scikit-learn',
        'statsmodels',
        'scipy',
        'requests'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    url='https://github.com/xu62u4u6/pymaftools',
)