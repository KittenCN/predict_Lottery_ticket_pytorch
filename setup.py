# -*- coding: utf-8 -*-
"""
Setup script for lottery prediction package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="predict-lottery-ticket-pytorch",
    version="1.0.0",
    author="KittenCN",
    author_email="",
    description="基于PyTorch的彩票预测系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KittenCN/predict_Lottery_ticket_pytorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lottery-get-data=scripts.get_data:main",
            "lottery-train=scripts.train_model:main", 
            "lottery-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml"],
    },
)