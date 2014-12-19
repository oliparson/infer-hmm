infer-hmm: An Infer.NET implementation of the hidden Markov model
==========

A [hidden Markov model](http://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) is a statistical tool for modelling sequential processes. The model consists of a Markov chain of discrete unobserved variables, each of which emits an observed continuous variable. The probabilistic dependencies between each of the variables in the chain is governed by a set of model parameters. Below is a graphical model of an HMM:
<img src="https://raw.githubusercontent.com/oliparson/infer-hmm/master/bayes-hmm.png" alt="Bayesian HMM graphical model" style="width: 25% height: 25%;"/>

This project provides a C# definition for a HMM using the [Infer.NET framework](http://research.microsoft.com/en-us/um/cambridge/projects/infernet/). This code makes it easy to run approximate Bayesian inference over both the model parameters and states of a HMM.

Special thanks go to Microsoft Research for adding support for chain models, and to [Matteo Venanzi](http://users.ecs.soton.ac.uk/mv1g10/) for his expertise in increasing the efficiency of the model.

##### Requirements

- .NET Framework 4.5

- Infer.NET 2.6
