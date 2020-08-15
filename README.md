# Tuned Gradient Descent with Momentum
(Code repositoy for the Master Thesis of Benjamin)

## Usage
To use the experiments install the package locally:

```bash
pip install -e .
```

## Gradient Descent with Momentum and Regularization
Gradient Descent with Momentum (GDM) is in general use for training
neural networks. The following hyperparameters have either to be set or
scheduled correctly: learning rate $\alpha$, momentum $\beta$ and the
amount of regularization $\lambda$.

## TGDM:
This repository provides the code for a self tuning version of GDM with the previous presented hyperparameters. We work in python and support pytorch.
Our approach is derived from hyperparameter optimization using T steps of the inner iterations before we update the hyperparameters.
We are Hessian free and require additional 1/T gradient evaluations.

### Hyperparameter Optimization (HO):
The general approach of gradient based Hyperarameter Optimization (HO) is as follows: We take the whole optimization trace and differentiate it in respect of a hyperparamter.
With this optained hypergradient we do then Gradient Descent on the hyperparameter and have Hyperparameter Optimization. Unfortunatly this is not feasible for neural networks.
In this repository we derived three different approximations: Hyper Descent (HD), T1-T2 and Ours.

### Hyper Descent TGDM (TGDM_HD/C):
This approach was presented by [1] and does use two sequential minibatches to perform HO. Usage:

```python
from tgdm import TGDM_HD, TGDM_HDC

# settings
defaults = {'hyper_learning_rate': [args.hlr_lr, args.hlr_momentum, args.hlr_regularization],\
            'lr': args.lr,\
            'momentum': args.momentum,\
            'regularization': args.regularization}
regulation = RegularizationFactory().by_name(args.regulation) #L2, L1, noise

# optimizer
optimizer = TGDM_HDC(model.parameters(), defaults, regulation)

#HD alternates internally between updating the model parameters and hyperparams:

for i in range(iters):
	model.train()
	optimzier.zero_grad()
	C = ... # compute cost with training data
	C.backward()
	optimizer.step() # done!
```

### T1-T2 TGDM (TGDM_T1T2):
The T1-T2 Method [2] uses a single trainig batch followed by a single validation batch to perform HO. This results in some significant overhead. Usage:

```python
from tgdm import TGDM_T1T2

optimizer = TGDM_T1T2(model.parameters(), defaults, regulation)

for i in range(iters):
	# prepare HO
	optimizer.hyper_zero_grad()
	# single inner step:
	model.train()
	optimizer.zero_grad()
	C = ... # compute cost with training data
	C.backward()
	optimizer.step()
	# single HO step:
	model.eval() # important!
	optimizer.zero_grad()
	E = ... # compute cost with validation data
	E.backward()
	optimizer.hyper_step()
```

### Own TGDM
Our own approximation performs first T inner optimization steps before we do HO. This reduces the overhead and allows better hypergradient estimates:

```python
from tgdm import TGDM

optimizer = TGDM(model.parameters(), defaults, regulation)

for i in range(iters):
	# prepare HO
	optimizer.hyper_zero_grad()	
	# perform T-steps of the inner optimization
	for t in range(T):
		model.train()		
		optimizer.zero_grad()
		C = ... # compute cost with training data
		C.backward()
		optimizer.step()
	# followed by one HO step:
	model.eval() # important!
	optimizer.zero_grad()
	E = ... # compute cost with validation data
	E.backward()
	optimizer.hyper_step()
```

## Literature
[1] Atilim Gunes Baydin, Robert Cornish, David Martinez Rubio, Mark Schmidt, and
Frank Wood. Online learning rate adaptation with hypergradient descent. arXiv
preprint arXiv:1703.04782, 2017.


[2] Jelena Luketina, Mathias Berglund, Klaus Greff, and Tapani Raiko. Scalable
gradient-based tuning of continuous regularization hyperparameters. In Interna-
tional conference on machine learning, pages 2952–2960, 2016.





