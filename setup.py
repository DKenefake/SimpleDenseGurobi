from setuptools import find_packages, setup

__version__ = "0.0.1"

setup(
    name="simpleDenseGurobi",
    author="Dustin Kenefake",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.stan"]}
)
