from setuptools import setup, find_namespace_packages

requirements = (
    "pandas",
    "xarray",
)

setup(
    name="bonner-computation",
    version="0.1.0",
    packages=find_namespace_packages(),
    install_requires=requirements,
)
