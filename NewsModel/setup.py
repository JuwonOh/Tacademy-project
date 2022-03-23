import io
import os
import sys

from setuptools import find_packages, setup

with open("README.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()

setup(
    name="Newspiece",
    version="1.4",
    description="Tacademy project의 newsmodel을 학습하고 실험하기 위한 패키지 입니다.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juwon Oh, Jinho Han",
    author_email="13a71032776@gmail.com",
    url="https://github.com/JuwonOh/Tacademy-project",
    download_url="https://github.com/JuwonOh/Tacademy-project/tree/main/NewsModel",
    license="MIT License",
    install_requires=[
        "torch==1.11.0",
        "mlflow==1.24.0",
        "scikit-learn==1.0.2",
        "transformers==4.0",
        "scipy==1.7.1",
    ],
    packages=find_packages(where="NewsModel"),
    keywords=["NLP"],
    python_requires=">=3.8",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
