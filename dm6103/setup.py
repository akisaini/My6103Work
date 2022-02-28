from setuptools import setup, find_packages

setup(
    name='dm6103',
    version='1.0.20220225',
    description='Python Package for George Washington University DATS 6103',
    packages=find_packages(),
    install_requires=[
        'pandas>=0.23.3',
        'requests>=2.18.4',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'mysql-connector-python>=8.0'
    ]
)
