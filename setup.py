import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="olpy",
    version="1.0.0.dev2",
    author="Boladji Vinny",
    author_email="vinny.adjibi@outlook.com",
    description="An online machine learning package for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boladjivinny/olpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={'': ['data/a1a*', 'data/svmguide*' ]},
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires = [
        'numpy>=1.20.1',
        'pandas>=1.1.3',
        'scikit-learn>=0.24.1',
    ]
)
