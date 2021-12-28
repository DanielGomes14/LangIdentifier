import argparse
from lang import Lang
from findlang import FindLang
from locatelang import LocateLang
import sys
import logging


class Main:
    def __init__(self) -> None:
        self.lang = None
        self.findlang = None
        self.locatelang = None
        ref_filename, dir_ref_files, target_filename, locate_lang,\
            compare_langs, k, alpha, multi_k, test_dir, threshold_alphabet =\
                self.check_arguments()

        if not dir_ref_files:
            self.lang = Lang(ref_filename, target_filename, k, alpha)
            self.lang.run()
            self.lang.bits_compress_target()
        elif locate_lang:
            self.locatelang = LocateLang(dir_ref_files, target_filename, k, alpha, multi_k, threshold_alphabet)
            self.locatelang.run(compare_langs)
        else:
            self.test_dir = test_dir
            self.findlang = FindLang(dir_ref_files, target_filename, k, alpha, test_dir)
            self.findlang.run()

        self.get_results()

    def get_results(self) -> None:
        if self.lang:
            print(f"Entropy of reference file {self.lang.fcm.entropy}")
            print(f"Total characters in target file {self.lang.t_number_chars}")
            print(f"Estimated average number of bits {self.lang.t_number_chars*self.lang.fcm.entropy}")
            print(f"Number of bits necessary to compress target file with a trained model {self.lang.n_bits}")
        if self.findlang:
            right_predictions = 0
            for i, item in enumerate(self.findlang.guessed_langs.items()):
                lang, guessed_lang = item
                if lang == guessed_lang:
                    right_predictions += 1
                print(f"\nFor {lang} file:")
                print(f"\t- Guessed language: {guessed_lang}")
                print(f"-Accuracy: {right_predictions/(i+1)}")
        if self.locatelang:
            [print(f"Position {pos}, language: {lang}") for pos, lang in self.locatelang.location_langs.items()]


    def usage(self):
        print("Usage: python3 main.py\
                \n\t-r <file name for reference file:str>\
                \n\t-d <directory name for reference files:str>\
                \n\t-t <file name for target file:str>\
                \n\t-td <directory name for target files:str>\
                \n\t-k <context size:int>\
                \n\t-a <alpha:int>\
                \n\t-l <use LocateLang module>\
                \n\t-m <multiple k values:int[]> Example: 2 3 4\
                \n\t-ta <threshold with alphabet size in consideration>\
                \n\t-c <use compareLang method>")


    def check_arguments(self):
        arg_parser = argparse.ArgumentParser(
            prog="Finite Context Model",
            usage=self.usage
        )
        arg_parser.add_argument('-r', nargs=1, default=["./../datasets/languages_train/English.utf8"])
        arg_parser.add_argument('-d', nargs=1, default=[None])
        arg_parser.add_argument('-t', nargs=1, default=["./../datasets/languages_test/English.utf8"])
        arg_parser.add_argument('-k', nargs=1, type=int, default=[3])
        arg_parser.add_argument('-a', nargs=1, type=float, default=[0.001])
        arg_parser.add_argument('-l', action='store_true')
        arg_parser.add_argument('-ta', action='store_true')
        arg_parser.add_argument('-c', action='store_true')
        arg_parser.add_argument('-m', nargs='*', type=int, default=[])
        arg_parser.add_argument('-td', nargs=1, default=[None])

        args = None

        try:
            args = arg_parser.parse_args()
        except:
            self.usage()
            sys.exit(0)

        ref_filename = args.r[0]
        dir_ref_files = args.d[0]
        target_filename = args.t[0]
        k = args.k[0]
        alpha = args.a[0]
        test_dir = args.td[0]

        return ref_filename, dir_ref_files, target_filename, args.l, args.c, k,\
            alpha, args.m, test_dir, args.ta


if __name__ =="__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main = Main()
