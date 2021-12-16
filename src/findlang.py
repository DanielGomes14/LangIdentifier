import logging
from fcm import FCM
from lang import Lang
import os
import sys
from utils import open_file, open_dir


class FindLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.1, test_dir=None) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.test_dir = test_dir
		if test_dir:
			self.test_dir = test_dir if test_dir[-1] == '/' else f"{test_dir}/"
		self.k = k
		self.alpha = alpha

		self.guessed_langs = {}


	def get_langs(self, target_filename=""):
		try:
			return [Lang(f"{self.dir_ref_files}{ref_file}", target_filename, self.k, self.alpha)\
				for ref_file in os.listdir(self.dir_ref_files)]
		except:
			logging.info(f"Directory {self.dir_ref_files} not found")
			sys.exit()


	def get_t_alphabet(self):
		if self.test_dir:
			target_files = open_dir(self.test_dir)
			t_alphabet = set()

			for target_filename in target_files:
				file_text = open_file(f"{self.test_dir}{target_filename}", "r")

				for line in file_text:
					for char in line:
						t_alphabet.add(char)

			return t_alphabet

		file_text = open_file(self.target_filename, "r")

		t_alphabet = set()

		for line in file_text:
			for char in line:
				t_alphabet.add(char)
		
		return t_alphabet


	def run(self):
		t_alphabet = self.get_t_alphabet()
		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		
		if self.test_dir:
			target_files = open_dir(self.test_dir)
			langs = self.get_langs("")
			[lang.run(t_alphabet) for lang in langs]

			for target_filename in target_files:
				f = open_file(f"{self.test_dir}/{target_filename}", "r")
				target_text = f.read()
				f.close()
				target_lang_name = target_filename.split('.')[-2].split('/')[-1]
				[lang.bits_compress_target(target_text) for lang in langs]
				self.guessed_langs[target_lang_name] = self.guess_language(langs)

			return

		f = open_file(self.target_filename, 'r')
		target_text = f.read()
		f.close()
		langs = self.get_langs(self.target_filename)
		target_lang_name = self.target_filename.split('.')[-2].split('/')[-1]
		[(lang.run(t_alphabet), lang.bits_compress_target(target_text)) for lang in langs]
		self.guessed_langs[target_lang_name] = self.guess_language(langs)


	def guess_language(self, langs):
		necessary_bits = [lang.n_bits for lang in langs]
		
		min_bits = min(necessary_bits)

		return langs[necessary_bits.index(min_bits)].lang_name
