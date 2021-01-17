import string
from json import dump, load

import numpy as np

from layers.GPTLayers import *

"""
This file contains implementations for a Language Model based on Transformers, as well as a TextGenerator that can use it.

Working tf version: 1.15.0
Author: Víctor Busqué
"""

class LanguageModel(tf.keras.Model):
    """ Model that enables text-generation and pre-training for language knowledge. It can be trained to do next-word prediction, masking future tokens from context.
            
        Attributes:
            max_seq_length (int): Maximum number of tokens in a sequence (it affects both the amount of context and the amount of words it can generate).
            vocab_size (int): Number of different words the model can learn (both in its input and its output).
            embed_dim (int): Dimensionality of the internal embedding vectors.
            num_heads (int): Number of different projections the model will use to represent each k,v,q vector. It must obey: embed_dim % num_heads == 0.
            feed_forward_dim (int): Dimensionality of the feed forward layer. Rule of thumb: feed_forward_dim = 4*embed_dim, as seen in the original paper.
            num_transformer_blocks (int): Amount of transformer blocks to use.
            embedding_layer (GPTLayers.TokenAndPositionEmbedding): Token and Position embedding layer.
            transformer_blocks(List[GPTLayers.TransformerBlock]): Stack of transformer blocks.
            output_layer(tf.keras.layers.Dense): Final fully connected layer.
    """

    def __init__(self, 
                config_path: str=None,
                max_seq_length: int=50,
                vocab_size: int=10000,
                embed_dim: int=256,
                num_heads: int=4,
                feed_forward_dim: int=1024,
                num_transformer_blocks: int=2):
        """Instantiates a Language Model

        Args:
            config_path (str, optional): Path to a JSON configuration file. Defaults to None.
            max_seq_length (int, optional): Maximum number of tokens in a sequence. Defaults to 50. Ignored if config_path is set.
            vocab_size (int, optional): Number of different words the model can learn. Defaults to 10000. Ignored if config_path is set.
            embed_dim (int, optional): Dimensionality of the internal embedding vectors. Defaults to 256. Ignored if config_path is set.
            num_heads (int, optional): Number of different projections the model will use to represent k,v,q vectors. Defaults to 4. Ignored if config_path is set.
            feed_forward_dim (int, optional): Dimensionality of the feed forward layer. Defaults to 4*embed_dim. Ignored if config_path is set.
            num_transformer_blocks (int, optional): Amount of transformer blocks to use. Defaults to 2. Ignored if config_path is set.
        """

        super(LanguageModel, self).__init__()

        if config_path:
            with open(config_path, "r", encoding="utf8") as f: config = load(f)

            self.max_seq_length = config["max_seq_length"]
            self.vocab_size = config["vocab_size"]
            self.embed_dim = config["embed_dim"]
            self.num_heads = config["num_heads"]
            self.feed_forward_dim = config["feed_forward_dim"]
            self.num_transformer_blocks = config["num_transformer_blocks"]     
        else:
            self.max_seq_length = max_seq_length
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.feed_forward_dim = feed_forward_dim
            self.num_transformer_blocks = num_transformer_blocks

        self.embedding_layer = TokenAndPositionEmbedding(self.max_seq_length, self.vocab_size, self.embed_dim)
        self.transformer_blocks = [TransformerBlock(self.embed_dim, self.num_heads, self.feed_forward_dim) for _ in range(self.num_transformer_blocks)]
        
        self.output_layer = layers.Dense(self.vocab_size, name="token_classifier")
        self.build( input_shape=(self.max_seq_length,) )
    
    def save_config(self, path: str) -> None:
        """Saves a JSON file with the configuration parameters to path.

        Args:
            path (str): Path where the JSON must be saved.
        """
        config = {
            "max_seq_length": self.max_seq_length, 
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "num_transformer_blocks": self.num_transformer_blocks
        }
        with open(path, "w", encoding="utf8") as f: dump(config, f, indent=4)

    def summary(self) -> None:
        """Prints a summary of the Language Model."""
        x = layers.Input(shape=(self.max_seq_length,), name="input_token_ids")
        p_mod = keras.Model(inputs=[x], outputs=self.call(x))
        p_mod.summary()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Performs a prediction for a given input.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Prediction.
        """
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        outputs = self.output_layer(x)
        return outputs


class TextGenerator(object):
    """Wrapper around Language Model to interact with text.

    Attribute:
        model (LanguageModel): Language Model that runs the Text Generator.
        word2id (Dict): Mapping from word -> ID.
        id2word (List): Mapping from ID -> Word. Just a list of words sorted by word ID.
        max_seq_length (int): Maximum length of a sentence.
    """

    def __init__(self, 
                model_weights_path:str, 
                vocabulary_path: str, 
                config_path: str):
        """Instantiates a TextGenerator

        Args:
            model_weights_path (str): Path to the ".h5" file where the language model weights have been saved.
            vocabulary_path (str): Path to the JSON file where the vocabulary has been saved.
            config_path (str): Path to the JSON file where the LanguageModel configuration has been saved.
        """
        self.model = LanguageModel(config_path=config_path)
        self.model.load_weights(model_weights_path)

        with open(vocabulary_path, "r", encoding="utf8") as f: 
            self.word2id = load(f)
        self.id2word = list(self.word2id.keys()) # {i: word for word,i in self.word2id.items()}

        self.max_seq_length = self.model.max_seq_length

        
    def preprocess(self, text: str) -> List[str]:
        """This function should pre-process text to the correct format. It needs to be implemented externally to fit into any scenario.

        Args:
            text (str): Text to be processed

        Raises:
            NotImplementedError: When it has not ben overriden.

        Returns:
            List[str]: Processed version of the input text. Needs to be tokenized (splitted).
        """
        raise NotImplementedError("This function needs to be implemented externally. Then do \"text_generator.preprocess = preprocess\"")

    def encode(self, tokens: List[str]) -> np.ndarray:
        """This function should encode text to the correct input vector.
        It performs standard encoding using word2id map. If it needs to be different, it should be overriden.

        Args:
            tokens (List[str]): Text to be encoded.

        Returns:
            Union[np.ndarray, List, tf.Tensor]: Array like of word_id for each word in the text
        """
        get_token_id = lambda token: self.word2id[token] if token in self.id2word else self.word2id["<unk>"]
        x = np.zeros((self.model.max_seq_length))
        x[0] = self.word2id["<start>"]
        for i, token in enumerate(tokens):
            x[i+1] = get_token_id(token)
        return x


    def sample_from(self, 
                    logits: tf.Tensor, 
                    top_p: float, 
                    top_k: int=10
                    ) -> int:
        """Samples a value from prediction logits based on parameters.

        Args:
            logits (tf.Tensor): Logits from the predictions (i.e. probability distribution).
            top_p (float): Cumulative probability to be considered.
            top_k (int, optional): Number of words to take into account. Defaults to 10.

        Returns:
            int: [description]
        """
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        
        p, i = preds[0], 1
        while p < top_p:
            p += preds[i]
            i += 1
        indices = indices[:i]
        preds = preds[:i]
        preds /= sum(preds)
        return np.random.choice(indices, p=preds)

    def predict_sentence(self, 
                        sentence: str, 
                        top_p: float, 
                        max_words: int, 
                        top_k: int=10,
                        unk_token: str="<unk>",
                        end_token: str="<end>"
                        ) -> str:
        """Generates a sentence.

        Args:
            sentence (str): Input sentence.
            top_p (float): Cumulative probability to be considered at each step.
            max_words (int): Maximum amount of words to be predicted.
            top_k (int, optional): Number of words to take into account at each step. Defaults to 10.
            unk_token (str, optional): Token defined for unknown tokens. Defaults to "<unk>".
            end_token (str, optional): Token defined as the end token. Defaults to "<end>".

        Returns:
            str: Completed text.
        """
        sentence = self.preprocess(sentence)

        unk_idx = self.word2id[unk_token]

        init_num_words = len(sentence.split())
        last_word = sentence.split()[-1]

        curr_word_index = init_num_words

        x = self.encode(sentence.split())
        while True:
            y = self.model.predict(np.array([x]))
            y = y[0][curr_word_index]
            y[unk_idx] = 0 # Masking <unk> token to 0 score, so that it is not elected

            word_id = self.sample_from(y, top_p, top_k)
            word = self.id2word[word_id]
            x[curr_word_index+1] = self.word2id[word]
                
            if word == end_token: break
            
            if not word in [",",".", ")",":"] and not last_word == "(": sentence += ' ' + word
            else: sentence +=  word
            
            last_word = word
            curr_word_index += 1

            if curr_word_index == min(self.max_seq_length-2 , init_num_words+max_words-1): break
        return sentence