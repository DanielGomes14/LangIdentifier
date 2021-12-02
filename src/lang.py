import logging
from fcm import FCM
from math import log2
import sys


class Lang:
    def __init__(self, reference_filename, target_filename, k=3, alpha=0.1) -> None:
        self.ref_filename = reference_filename
        self.target_filename = target_filename
        self.k = k
        self.alpha = alpha
        self.fcm = FCM(k,alpha,reference_filename)

        self.t_number_chars = 0 # number of chars in target file
        self.n_bits = 0 # number of bits to compress the text

        self.lang_name = self.ref_filename.split('.')[-2].split('/')[-1]
    
    
    def train_fcm(self):
        logging.info(f"Starting to train FCM with {self.lang_name}")
        self.fcm.run(get_alphabet=False)

    
    def merge_alphabets(self, t_alphabet=None):
        max_index = max(self.fcm.alphabet.values())

        if not t_alphabet:
            try:
                file_text = open(self.target_filename, "r", encoding='utf-8')
            except FileNotFoundError:
                print(f"Could not open file {self.target_filename}")
                sys.exit(0)

            for line in file_text:
                for char in line:
                    self.t_number_chars += 1
                    if char not in self.fcm.alphabet:
                        max_index += 1
                        self.fcm.alphabet[char] = max_index
                        self.fcm.alphabet_size += 1
                        logging.info(f'Adding char {char} to reference alphabet')
        else:
            for char in t_alphabet:
                if char not in self.fcm.alphabet:
                    max_index += 1
                    self.fcm.alphabet[char] = max_index
                    self.fcm.alphabet_size += 1
                    logging.info(f'Adding char {char} to reference alphabet')


    def run(self, t_alphabet=None):
        self.fcm.read_file()

        self.merge_alphabets(t_alphabet)
        
        self.train_fcm()

    
    def bits_compress_target(self, target_text=None):
        # logging.info(f"Calculating number of bits to compress with a trained FCM with {self.lang_name}.")
        self.n_bits = 0

        if not target_text:
            try:
                f = open(self.target_filename, 'r', encoding='utf-8')
            except FileNotFoundError:
                logging.error(f"Could not open file {self.target_filename}")
                sys.exit()
            target_text = f.read()
        

        context = target_text[:self.k]

        for next_char in target_text[self.k:]:
            context_probabilities = self.fcm.get_context_probabilities(context)

            next_char_index = self.fcm.alphabet[next_char]
            self.n_bits -= log2(context_probabilities[next_char_index])

            context = context[1:] + next_char

        # logging.info(f"The number of bits necessary are {self.n_bits}")

        return self.n_bits