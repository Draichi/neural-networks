# Neural networks
>From randomness to score 162
<img src="ai.gif">

## Requirements
- [Anaconda](https://conda.io/docs/user-guide/install/index.html)
- Tensorflow
- [OpenAI gym](https://github.com/openai/gym/blob/master/README.rst#installing-everything)
- [TFlearn](http://tflearn.org/installation/)

## Setup
```sh
conda install -c anaconda tensorflow-gpu

pip install tflearn

apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

cd ~/
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[all]'
```
