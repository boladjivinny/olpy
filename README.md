**OLPy** is a Python module for classification using online machine learning 
models. It is built using an interface similar to that of **scikit-learn** 
allowing users to use it right away for their various relevant tasks.

Website: https://olpy.readthedocs.io/en/latest/

# Installation

## Dependencies
OLPy requires:

- Python (>=3.6)
- Numpy  (>= 1.20.1)
- scikit-learn (>=0.24.1)

## User installation
The easiest way to install olpy is using `pip`

```
pip install -U olpy
```

## Usage
OLPy comes with a main application that allows users to run a test given a 
train and testing dataset. The basic syntax for using the script is:

```
python3 -m olpy -s <random seed> -l <label of y in the dataset> <training data> <testing data>
```

By default, it expects the label field to be set to `Label` and the expected
format when running the script is CSV.

For example, the following command runs a test using the svm3guide3 dataset.
Before running it, make sure to download the relevant data from the `olpy/datasets/data` folder in the repository:

```
python3 -m olpy -s 32 -l 0 svmguide3 svmguide3.t
```

This prints the following table with a set of metrics to evaluate the
performances of the models on the given dataset.

```
algorithm	train time (s)	test time (s)	accuracy	f1-score	 roc-auc	true positive	true negative	false positive	false negative

scw2        	0.007872	0.000014	0.268293	0.423077	nan  	0.268293	0.000000	0.000000	0.731707
cw          	0.026443	0.000015	0.219512	0.360000	nan  	0.219512	0.000000	0.000000	0.780488
pa2         	0.042131	0.000014	0.365854	0.535714	nan  	0.365854	0.000000	0.000000	0.634146
pa          	0.043486	0.000014	0.365854	0.535714	nan  	0.365854	0.000000	0.000000	0.634146
arow        	0.043447	0.000014	0.341463	0.509091	nan  	0.341463	0.000000	0.000000	0.658537
pa1         	0.018348	0.000025	0.170732	0.291667	nan  	0.170732	0.000000	0.000000	0.829268
aromma      	0.026140	0.000014	0.097561	0.177778	nan  	0.097561	0.000000	0.000000	0.902439
iellip      	0.026845	0.000014	0.243902	0.392157	nan  	0.243902	0.000000	0.000000	0.756098
romma       	0.140190	0.000013	0.219512	0.360000	nan  	0.219512	0.000000	0.000000	0.780488
narow       	0.009500	0.000014	0.243902	0.392157	nan  	0.243902	0.000000	0.000000	0.756098
alma        	0.009521	0.000013	0.243902	0.392157	nan  	0.243902	0.000000	0.000000	0.756098
scw         	0.010670	0.000015	0.243902	0.392157	nan  	0.243902	0.000000	0.000000	0.756098
perceptron  	0.003107	0.000013	0.243902	0.392157	nan  	0.243902	0.000000	0.000000	0.756098
ogd         	0.023205	0.000015	0.000000	0.000000	nan  	0.000000	0.000000	0.000000	1.000000
nherd       	0.013958	0.000014	0.560976	0.718750	nan  	0.560976	0.000000	0.000000	0.439024
sop         	0.019392	0.000016	0.560976	0.718750	nan  	0.560976	0.000000	0.000000	0.439024
```

A detailed documentation for the package is available at https://olpy.readthedocs.io/en/latest/.

# Contributing
At this stage, the project welcomes contributions in the following aspects:

- Unit testing
- Adding more models
- Improving the documentation
- Extending the models to do regression tasks as well where possible

The algorithms currently implemented are:

- Perceptron: the classical online learning algorithm (Rosenblatt, 1958);
- ALMA: A New Approximate Maximal Margin Classification Algorithm Gentile (2001);
- ROMMA: the relaxed online maxiumu margin algorithms (Li and Long, 2002);
- OGD: the Online Gradient Descent (OGD) algorithms (Zinkevich, 2003);
- PA: Passive Aggressive (PA) algorithms (Crammer et al., 2006);
- SOP: the Second Order Perceptron (SOP) algorithm (Cesa-Bianchi et al., 2005);
- CW: the Confidence-Weighted (CW) learning (Dredze et al., 2008);
- IELLIP: online learning algorithms by improved ellipsoid method Yang et al. (2009);
- AROW: the Adaptive Regularization of Weight Vectors (Crammer et al., 2009);
- NAROW: New variant of Adaptive Regularization (Orabona and Crammer, 2010);
- NHERD: the Normal Herding method via Gaussian Herding (Crammer and Lee, 2010)
- SCW: the recently proposed Soft ConfidenceWeighted algorithms (Wang et al., 2012).
- SCW2: Soft ConfidenceWeighted version 2 (Wang et al., 2012).


# Getting help
To get support regarding this package, please log an issue or shoot me an email
at vinny.adjibi@outlook.com and I will make sure to answer as soon as possible.
