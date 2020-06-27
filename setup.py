import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modelcreator",
    version="0.9.0",
    author="Bartłomiej Pogodziński",
    author_email="bartek.pogod@gmail.com",
    description="Machine Learning package for quick fast model generation and comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BartekPog/AutoML-Library",
    packages=setuptools.find_packages(exclude=["tests*"]),
    license='MIT',
    python_requires='>=3.6',
    install_requires=['joblib', 'sklearn', 'pandas'],
)
