from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

with open("./readme.md", "r", encoding="UTF8") as fh:
    long_description = fh.read()
    fh.close()



setup(
    name="newsmodel",
    version="0.26",
    description="Tacademy project의 newsmodel을 학습하고 실험하기 위한 패키지 입니다.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Juwon Oh, Jinho Han",
    author_email="13a71032776@gmail.com",
    url="https://github.com/JuwonOh/Tacademy-project",
    download_url="https://github.com/JuwonOh/Tacademy-project/tree/main/NewsModel",
    license="MIT License",
    install_requires=[
        "torch>=1.8.2+cpu",
        "mlflow>=1.24.0",
        "scikit-learn>=1.0.2",
        "transformers>=4.16.2",
        "nltk",
        "progressbar",
    ],
    packages=find_packages(),  # 이것만 적으면 만들때 특정한 것을 포함, exclude도 가능.
    py_modules=[
        splitext(basename(path))[0] for path in glob("newsmodel/*.py")
    ],
    python_requires=">=3.8",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
