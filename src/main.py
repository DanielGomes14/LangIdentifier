import argparse
from lang import Lang
from findlang import FindLang
import sys
import logging

FILENAME = "./../example/example.txt"

class Main:
    def __init__(self) -> None:
        self.lang = None
        self.findlang = None
        
        ref_file_names, target_filename, k, alpha = self.check_arguments()

        if len(ref_file_names) == 1:
            self.lang = Lang(ref_file_names[0], target_filename, k, alpha)
            self.lang.run()
        else:
            self.findlang = FindLang(ref_file_names, target_filename, k, alpha)
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
        arg_parser.add_argument('-r', nargs='*', default=[FILENAME])
        arg_parser.add_argument('-t', nargs=1, default=[FILENAME])
        arg_parser.add_argument('-k', nargs=1, type=int, default=[3])
        arg_parser.add_argument('-a', nargs=1, type=float, default=[0.1])
        args = None

        try:
            args = arg_parser.parse_args()
        except:
            self.usage()
            sys.exit(0)

        ref_file_names = args.r
        target_filename = args.t[0]
        k = args.k[0]
        alpha = args.a[0]

        return ref_file_names, target_filename, k, alpha


if __name__ =="__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main = Main()