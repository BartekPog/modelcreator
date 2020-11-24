import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelcreator",
    version="0.9.3",
    author="Bartłomiej Pogodziński",
    author_email="bartek.pogod@gmail.com",
    description="Machine Learning package for quick fast model generation and comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BartekPog/modelcreator",
    packages=setuptools.find_packages(exclude=["tests*"]),
    license='MIT',
    python_requires='>=3.6',
    install_requires=['joblib==0.15.0', 'scikit-learn==0.23.1',
                      'pandas==1.0.3', 'dask-ml==1.5.0', 'dask==2.19.0', 'imbalanced-learn==0.7.0'],
)
