This respository has become a place where I experiement with different frameworks and techniques.

I've tried:
* Keras / Tensorflow
* PytorchLightning / Pytorch
* Haiku / Jax

They each have their good points. so far, Keras and Haiku are my favourites. Pytorch feels very clunky and unintuitive to me. 

Bus prediciton itself has taken a back seat for me with this one, but I'd like to get my error lower.

Sometimes JAX doesn't kill the process properly if I KeyboardInterrupt my way out. The solution I've found is to: 
* sudo fuser -v /dev/nvidia*
* sudo kill -9 PID

or just
* killall run.py

