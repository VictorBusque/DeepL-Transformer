import string

import numpy as np

from models.TransformerLM import LanguageModel, TextGenerator

if __name__ == "__main__":
    import tensorflow.keras.backend as K

    RUN_MODE = "Simulator" # "Generator", "Simulator"
    
    if RUN_MODE == "Simulator":
        """
        Model instantiation simulator
        
        Here you can play with different parameters to see how the resulting model is.
        The example below is using standard GPT2-small parameters
        """
        model = LanguageModel(
            max_seq_length=1024,
            vocab_size=50_257,
            embed_dim=768,
            num_heads=12,
            feed_forward_dim=4*768,
            num_transformer_blocks=12     
        )
        model.summary()
        
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        params = trainable_count + non_trainable_count
        print("Model weights will weigh: "+ str(round((params*4)/(1024**2), 2)) + "Mb")
    else:
        """
        Pre-trained LM based on 25k spanish Wikipedia articles (about 100k sentences, ~5M tokens).
        It is word-based (11k dictionary), not using BPE or similar.
        
        You can check more about it on examples/wikipedia-8M-10e/model/ path.
        """
        text_generator = TextGenerator(
            model_weights_path="examples/wikipedia-8M-10e/model/wikipedia_lm.h5",
            vocabulary_path="examples/wikipedia-8M-10e/model/dictionary.json",
            config_path="examples/wikipedia-8M-10e/model/model_config.json"
        )
        print(text_generator.model.summary())
        
        def preprocess_sentence(sentence, add_start=True, add_end=True):
            punct = string.punctuation+'¿'
            sentence = sentence.lower()
            sentence = sentence.translate(str.maketrans({key: " {0} ".format(key) for key in punct}))
            sentence = sentence.replace('  ',' ')
            if add_start: sentence = "<start>" + sentence
            if add_end: sentence = sentence + "<end>"
            return sentence.strip()
        
        text_generator.preprocess = lambda t: preprocess_sentence(t, add_start=False, add_end=False)
        
        q = input("> ")
        while q != "quit":
            completion = text_generator.predict_sentence(q, max_words=120, top_p=.8)
            q = input(completion+"\n> ")
            
            
