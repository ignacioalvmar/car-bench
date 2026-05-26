from setuptools import find_packages, setup

setup(
    name="car_bench",
    version="0.1.0",
    description="The Car-Bench package",
    long_description=open("README.md").read(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "annotated_types>=0.7.0",
        "datasets>=3.0.0",
        "huggingface_hub>=0.25.0",
        "litellm>=1.80.16",
        "networkx>=3.1",
        "numpy>=2.0.0",
        "pandas>=2.2.0",
        "pydantic>=2.9.0",
        "pyvis>=0.3.2",
        "requests>=2.31.0",
        "seaborn>=0.13.0",
        "setuptools>=65.5.0",
        "tiktoken>=0.7.0",
        "tqdm>=4.66.0",
    ],
)
