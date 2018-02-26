# fproject Sentiment API

# Arquitecture
![Sentiment Api](https://imgur.com/v8emrBs.png)

# Tensorflow Serving 
Ones we have the trained model we need to load the model in the Tensorflow Serving Api and it will produce a gRPC server that we and interact with.

It's a little bit tricky and i recomend to use this tutorials:

- [How to Deploy a Tensorflow Model to Production by Siraj Raval](https://www.youtube.com/watch?v=T_afaArR0E8&t=1949s)

- [How to deploy Machine Learning models with TensorFlow by Vitaly Bezgachev](https://medium.com/towards-data-science/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7)

# gRPC Api
Now that we have our tensorflow model handle by the TF Api we have to make a client to talk to it.

## Dependencies

- python 3.6
- gensim
- tensorflow
- nltk
- grpcio 
- grpcio-tools

#### Download spanish word2vec model:

```sh
$ wget http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
$ gunzip SBW-vectors-300-min5.bin.gz
```

## Simple set up

- Modify the Tensorflow Api host and port values

## Usage

Clone this project.
```sh
$ cd fproject
$ python api.py
```


## Full System Arquitecture
![System](https://i.imgur.com/APqa7i0.png)

## NODE SERVER
* https://github.com/si-m/fproject_node


## FRONT-END REACT APP

* https://github.com/si-m/fproject_front
