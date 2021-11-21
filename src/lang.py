import logging
from fcm import FCM
import math

class Lang:
    def __init__(self, reference_filename, target_filename, k=3, alpha=0.1) -> None:
        self.ref_filename = reference_filename
        self.target_filename = target_filename
        self.k = k
        self.alpha = alpha
        self.fcm = FCM(k,alpha,reference_filename)
        self.n_bits = 0 # number of bits to compress the text
    
    
    def train(self):
        logging.info(f"Starting to train FCM with file {self.ref_filename}")
        self.fcm.run()

    
    def bits_compress_target(self):
        logging.info(f"Calculating number of bits to compress {self.ref_filename} with a trained FCM")

        with open(self.target_filename, 'r') as f:
            target_text = f.read()
            
            context = target_text[:self.k]
        
            for next_char in target_text[self.k:]:
                context_probabilities = self.fcm.get_context_probabilities(context)
                print(self.fcm.alphabet)
                print(next_char)
                next_char_index = self.fcm.get_context_index(next_char)

                self.n_bits -= math.log10(context_probabilities[next_char_index])

                context = context[1:] + next_char

        logging.info(f"Finished calculating. The number of bits necessary are {self.n_bits}")


