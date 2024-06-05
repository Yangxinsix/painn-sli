from setuptools import setup

setup(
    name="PaiNN",
    version="1.0.0",
    description="Library for implementation of message passing neural networks in Pytorch",
    author="xinyang",
    author_email="xinyang@dtu.dk",
    url = "https://github.com/Yangxinsix/PaiNN-model",
    packages=["PaiNN"],
    install_requires=[
        'ase>=3.22.1',
        'numpy',
        'asap3>=3.12.8',
        'torch>=1.10',
        'scipy',
    ]
)
