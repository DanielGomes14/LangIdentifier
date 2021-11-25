import argparse
from lang import Lang
from findlang import FindLang
import sys
import logging


class Main:
    def __init__(self) -> None:
        self.lang = None
        self.findlang = None
        
        ref_filename, dir_ref_files, target_filename, k, alpha = self.check_arguments()

        if not dir_ref_files:
            self.lang = Lang(ref_filename, target_filename, k, alpha)
            self.lang.run()
        else:
            self.findlang = FindLang(dir_ref_files, target_filename, k, alpha)
            self.findlang.run_langs()
            self.findlang.guess_language()

        self.get_results()


    def get_results(self) -> None:
        if self.lang:
            print(f"Entropy of reference file {self.lang.fcm.entropy}")
            print(f"Total characters in target file {self.lang.t_number_chars}")
            print(f"Estimated average number of bits {self.lang.t_number_chars*self.lang.fcm.entropy}")
            print(f"Number of bits necessary to compress target file with a trained model {self.lang.n_bits}")
        if self.findlang:
            [print(f"Number of bits for {lang.ref_filename}: {lang.n_bits}") for lang in self.findlang.langs]
            print(f"Guessed Language: {self.findlang.language}")


    def usage(self):
        print("Usage: python3 main.py\
                \n\t-r <file name for reference file:str>\
                \n\t-t <file name for target file:str>\
                \n\t-k <context size:int>\
                \n\t-a <alpha:int>\n")


    def check_arguments(self):
        arg_parser = argparse.ArgumentParser(
            prog="Finite Context Model",
            usage=self.usage
        )
        arg_parser.add_argument('-r', nargs=1, default=["./../datasets/language_train/english.utf8"])
        arg_parser.add_argument('-d', nargs=1, default=None)
        arg_parser.add_argument('-t', nargs=1, default=["./../datasets/language_test/english.utf8"])
        arg_parser.add_argument('-k', nargs=1, type=int, default=[3])
        arg_parser.add_argument('-a', nargs=1, type=float, default=[0.1])
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

        return ref_filename, dir_ref_files, target_filename, k, alpha


if __name__ =="__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main = Main()