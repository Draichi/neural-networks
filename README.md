# Neural networks

Each file is a individual implementations of a machine learning algorithm.

### Flappy Bird

Open `/flappy-bird/index.html`. or see the live [demo](https://draichi.github.io/ai-flappy-bird/index.html).

## Setup

```sh

sudo apt-get install -y python3-numpy python3-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig

conda create -f UBUNUT_GPU.yml

```

### Run

```sh
python mnist_convnet.py
```
and in another termnial:
```sh
tensorboard --logdir=logs/conv
```