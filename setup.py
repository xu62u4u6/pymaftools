from setuptools import setup, find_packages

setup(
    name='pymaftools',
    version='0.1',
    author = "xu62u4u6",
    author_email="199928ltyos@gmail.com",
    description='pymaftools is a Python package for handling and analyzing Mutation Annotation Format (MAF) files. It provides utilities for data manipulation and visualization, including classes for MAF parsing and oncoplot generation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'pandas', 
        'numpy', 
        'matplotlib', 
        'seaborn', 
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    url='https://github.com/xu62u4u6/pymaftools',
)