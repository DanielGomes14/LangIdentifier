import logging
from fcm import FCM
from lang import Lang
import os
import sys


class FindLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.1) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.k = k
		self.alpha = alpha

		self.langs = self.get_langs()

		self.language = ''


	def get_langs(self):
		try:
			return [Lang(f"{self.dir_ref_files}{ref_file}", self.target_filename,self.k, self.alpha)\
				for ref_file in os.listdir(self.dir_ref_files)]
		except:
			logging.info(f"Directory {self.dir_ref_files} not found")
			sys.exit()


	def get_t_alphabet(self):
		try:
			file_text = open(self.target_filename, "r", encoding='utf-8')
		except FileNotFoundError:
			print(f"Could not open file {self.target_filename}")
			sys.exit(0)

		t_alphabet = set()

		for line in file_text:
			for char in line:
				t_alphabet.add(char)
		
		return t_alphabet


	def run(self):
		t_alphabet = self.get_t_alphabet()

		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		[(lang.run(t_alphabet), lang.bits_compress_target()) for lang in self.langs]
		
		logging.info(f"Finished calculating the number of bits necessary with all reference files.")
		self.guess_language()
 

	def guess_language(self):
		necessary_bits = [lang.n_bits for lang in self.langs]
		
		min_bits = min(necessary_bits)
		logging.info(f"Finished calculating minimum value of bits")

		self.language = self.langs[necessary_bits.index(min_bits)].lang_name
		logging.info(f"Guessed language: {self.language}")
