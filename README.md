# Transformer based English to German translation

Fun project in which English to German translation was achieved using a model with custom transformer-based architecture.

### Source code folder structure:
- __model.py__ - contains a from-scratch implementation of a transformer-based model
- __utils.py__ - contains definitions of aux classes and functions such as NLPDataset and pad_collate_fn
- __train.py__ - contains a training loop with defined hiperparameters

### Pytorch modules implemented:
- PositionalEncoding
- PositionWiseFFN
- AddAndNorm
- EncoderModule
- Encoder
- DecoderModule
- Decoder
- Transformer

### (important) aux functions implemented:
- pad_collate_fn

### Dataset used
- Dataset used is a Flickr30k containing 30 000 english and german image labels.