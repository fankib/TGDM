# Tuned Gradient Descent with Momentum
(Code repositoy for the Master Thesis of Benjamin)

## Gradient Descent with Momentum and Regularization
Gradient Descent with Momentum (GDM) is in general use for training
neural networks. The following hyperparameters have either to be set or
even scheduled correctly: learning rate $\alpha$, momentum $\beta$ and the
amount of regularization $\lambda$.

## Tuned GDM (TGDM):
This repository provides the code for a self tuning version of GDM with the previous presented hyperparameters. We work in python and support pytorch.
Our approach is derived from hyperparameter optimization.

### Hyperparameter Optimization (HO):
The general approach in HO is simple to describe: We take the whole optimization trace and differentiate it in respect of a hyperparamter. With this optained hypergradient we do then Gradient Descent on the hyperparameter and have Hyperparameter Optimization. Unfortunatly this is not feasible for neural networks.
So we derived three different variants.

### Hyperdescent TGDM (HD-TGDM):
This approach was invented by ... and does use two sequential minibatches to perform HO. Usage:

```python
import TGDM.optim as optim

optimizer = optim.HD_TGDM(model.parameters(), hyper_learning_rate=[0.01, 0.01, 0.0001]) # hyper learning rates
optimizer.init(alpha=0.01, beta=0.8, regularization=0.01) # for initialization
optimizer.init_random() # for random initialization of the hyperparameters

#HD-TGDM alternates internally inbetween optimization and HO:

for i in range(iters):
	optimzier.zero_grad()
	L = ... # compute loss with training data
	L.backward()
	optimizer.step() # done!
```

### T1-T2 TGDM (T1T2-TGDM):
T1-T2 uses a single trainig batch followed by a single validation batch to perform HO. This results in some significant overhead. Usage:

```python
import TGDM.optim as optim

optimizer = optim.T1T2_TGDM(model.parameters, ...)
optimizer.init_random() # optional

for i in range(iters):
	# single inner step:
	optimizer.zero_grad()
	L = ... # compute loss with training data
	L.backward()
	optimizer.step()
	# single HO step:
	optimizer.hyper_zero_grad()
	E = ... # compute loss with validation data
	E.backward()
	optimizer.hyper_step()
```

### Own TGDM
Our own approximation performs first T inner optimization steps before we do HO. This reduces the overhead and allows better hypergradient estimates:

```python
import TGDM.optim as optim

optimizer = optim.TGDM(model.parameters(), ...)
optimizer.init_random() # optional

for i in range(iters):
	# perform T-steps of the inner optimization
	for t in range(T):
		data_train, target_train = next(train_loader)
		optimizer.zero_grad()
		L = ... # compute loss with training data
		L.backward()
		optimizer.step()
	# followed by one HO step:
	optimizer.hyper_zero_grad()
	E = ... # compute loss with validation data
	E.backward()
	optimizer.hyper_step()
```





