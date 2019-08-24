# HyperGAN Extensions

This project was done for Advanced Big Data Analysis, Math 389L at CGU. 
We worked on extending [HyperGAN](https://arxiv.org/abs/1901.11058).
In particular, we introduced:

* A dynamic hypernet constructor that can produce hypernets for arbitrary target networks, allowing us to experiment on a much larger ResNet20 target network.
* Training discrete target networks from a minibatch instead of treating them as a minibatch sized ensemble.
* Pretraining for the encoder and generators, to ensure that the initial distribution of weights breaks symmetry in multiple directions.
