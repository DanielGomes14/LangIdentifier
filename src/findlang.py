import logging
from fcm import FCM
from lang import Lang


class FindLang:
    def __init__(self, reference_filenames, target_filename, k=3, alpha=0.1) -> None:
        self.reference_filenames = reference_filenames
        self.target_filename = target_filename
        self.k = k
        self.alpha = alpha
        self.langs = [Lang(ref_file,target_filename,k,alpha) for ref_file in reference_filenames]

        self.language = ''
    
    
    def train(self):
        logging.info(f"Starting to train FCM with files {self.reference_filenames}")
        [lang.train() for lang in self.langs]
 

    def guess_language(self):
        necessary_bits = [lang.bits_compress_target() for lang in self.langs]
        logging.info(f"Finished calculating the number of bits necessary with all reference files.")
        
        min_bits = min(necessary_bits)
        logging.info(f"Finished calculating minimum value of bits")

        self.language = self.langs[necessary_bits.index(min_bits)].ref_filename
        logging.info(f"Guessed language: {self.language}")
