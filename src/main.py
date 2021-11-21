import argparse
from lang import Lang
import sys
import logging

FILENAME = "./../example/example.txt"

class Main:
    def __init__(self) -> None:
        ref_file_name, target_filename, k, alpha = self.check_arguments()
        self.lang = Lang(ref_file_name, target_filename, k, alpha)
        self.lang.train()
        self.lang.bits_compress_target()

        self.get_results()


    def get_results(self) -> None:
        print(f"Entropy of reference file {self.lang.fcm.entropy}")
        print(f"Number of bits necessary to compress target file with a trained model {self.lang.n_bits}")


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
        arg_parser.add_argument('-r', nargs=1, default=[FILENAME])
        arg_parser.add_argument('-t', nargs=1, default=[FILENAME])
        arg_parser.add_argument('-k', nargs=1, type=int, default=[3])
        arg_parser.add_argument('-a', nargs=1, type=float, default=[0.1])
        args = None

        try:
            args = arg_parser.parse_args()
        except:
            self.usage()
            sys.exit(0)

        ref_file_name = args.r[0]
        target_filename = args.t[0]
        k = args.k[0]
        alpha = args.a[0]

        return ref_file_name, target_filename, k, alpha


if __name__ =="__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main = Main()