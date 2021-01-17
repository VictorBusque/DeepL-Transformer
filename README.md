# DeepL-Transformer
This implementation is a slightly modified version of the transformer architecture as seen [here](https://keras.io/examples/generative/text_generation_with_miniature_gpt/), implemented using tensorflow-keras.
## Structure
### Layers
On `layers/GPTLayers.py` you can find an implementation of:
- **TokenAndPositionEmbedding**: `tf.keras.layer.Layer` that combines both an embedding layer with the position embedding required for transformers to keep track of word location. The position embedding is trained, not calculated using a formula based on sine and cosine as proposed on the original Transformer described [here](https://arxiv.org/abs/1706.03762). It is the input layer to the transformer.
- **MultiHeadSelfAttention**: `tf.keras.layer.Layer` that implements multi-head self-attention using Scaled Dot-Product attention, as seen here: <img src=https://www.researchgate.net/profile/Dennis_Gannon/publication/339390384/figure/fig2/AS:860759328317440@1582232424204/Multi-Head-Attention-and-Scaled-Dot-Product-Attention-again-from-Viswani-et-al.jpg>
  It accepts a `use-mask` parameter to turn it into masked multi-head self-attention (as used in models such as GPT2).
- **TransformerBlock**: `tf.keras.layer.Layer` that represents a Transformer Decoder block (without the typical encoder-decoder attention, since we won't use an encoder). This Layer uses the`MultiHeadSelfAttention` layer to implement the whole transformer decoder.

### Models
On `models/TransformerLM.py` you can find two classes:
- **LanguageModel**: `tf.keras.models.Model` that uses the previously described layers among with some other layers to represent a model that can be used for causal language modeling (CLM, or next word prediction). It basically combines a `TokenAndPositionEmbedding` among with an arbitrary number of `TransformerBlock` layers. It pushes the context vector from the last TransformerBlock through a Linear layer to be later transformed to a word probability distribution via softmax.
The result is the following: <img src=imgs/transformer-decoder.png>
- **TextGenerator**: It's a simple utility class that aims to make text generation and model interaction easier. It takes in the model's weights file (.h5), the model configuration file (a .json file that can be obtained by using `LanguageModel.save_config(path)`), and the vocabulary file (a .json file that maps word->word_id).
`run.py` aims to be an example on how to use these files and classes to do text generation.

It's important to notice that TextGenerator has two functions that may need to be overridden (`TextGenerator.preprocess` and `TextGenerator.encode`) to work according yo your dataset and language modeling needs.

## How to use it
The main idea of this "framework" is to be an easy-to-plug solution to train language models based on transformer's architecture. Now, the dataset cleaning, preprocessing and vocabulary generation needs to be done independently outside.

The ideal steps for using this would be:

1) Load the data.
2) Pre-process / clean the data (you can use BPE tokenizers or whatever).
3) Generate a word vocabulary map (word -> word_id) and **save it as a json dictionary**.
4) Encode the data (using the word vocabulary from the previous step) to generate training matrix `x`.
5) Generate labels by shifting the encoded data 1 step to the future (so `y_i[t] = x_i[t+1]`).
6) Instantiate a LanguageModel with some parameters. for example:<br />
   `model = LanguageModel(config_path = None, max_seq_length = 256, vocab_size = 25000, embed_dim = 256, num_heads = 4, feed_forward_dim = 4*256, num_transformer_blocks = 2)`
   <br/>Also **remember to save it's configuration by doing `model.save_config(path)`**.
7) Compile the model as you would in any other Tensorflow model. Usually something like:<br/> `model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")`
8) Fit the model with your data.
9) **Save the model weights by doing `model.save_weights(path)`**.

And that's it, with the 3 files saved you can Instantiate a `TextGenerator` to do some sentence predictions like this:<br/>
`text_generator = TextGenerator(model_weights_path="path/to/model_file.h5", vocabulary_path=path/to/dictionary.json", config_path="path/to/model_config.json")`
<br/>Override the `TextGenerator.preprocess` function so that sentences are preprocessed the same way they were preprocessed for training. If necessary, override the `TextGenerator.encode` function for the same purpose.<br/>Then, you're ready to go predicting sentences with your new transformer's based language model!.

## Example
I have personally used this to train some language models, and here I share an example of a language model trained on a spanish dataset i scrapped from wikipedia. The dataset is quite modest, it contains about 25.000 wikipedia articles, for a total of aprox. 150.000 sentences and ~5.000.000 tokens.

I decided to cut vocabulary on about ~11.000 total words (all the words that appeared more than 20 times among all articles). The maximum sequence length was set at 70 arbitrarily after checking a distplot on how it was distributed (it only truncated about 0.7% of the sentences to that size, the rest was smaller).

Training took about 35 minutes on an RTX 2070.

You can test the result by running the `run.py` command.

### Disclaimer
As said at the beginning, most of the implementation is based on code given on a keras tutorial, and I can't assure anyone everything is correct and optimized.

I tried to make most of the classes and functions readable and well-documented, but there might be mistakes here and there.