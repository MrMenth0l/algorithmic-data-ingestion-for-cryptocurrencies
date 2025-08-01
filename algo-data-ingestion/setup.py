from setuptools import setup, find_packages

setup(
    name="algo_data_ingestion",
    version="0.1.0",
    packages=find_packages(where="."),  # finds app/ and any other packages
)