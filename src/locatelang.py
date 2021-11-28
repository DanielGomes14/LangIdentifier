import logging
from math import log
from fcm import FCM
from lang import Lang
import os
import sys


class LocateLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.1) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.k = k
		self.alpha = alpha

		self.langs = self.get_langs()

		self.chunk_lang = {}

		self.CHUNK_SIZE = 10_000
		self.THRESHOLD = 0.02


	def get_langs(self):
		try:
			return [Lang(f"{self.dir_ref_files}{ref_file}", self.target_filename, self.k, self.alpha)\
				for ref_file in os.listdir(self.dir_ref_files)]
		except:
			logging.info(f"Directory {self.dir_ref_files} not found")
			sys.exit()



	def guess_language(self, text, last_lang, ignore_lang=False):
		if not last_lang or ignore_lang:
			necessary_bits = [lang.bits_compress_target(text) if lang != last_lang else lang.n_bits for lang in self.langs]
			min_bits = min(necessary_bits)
			return self.langs[necessary_bits.index(min_bits)], min_bits

		necessary_bits = last_lang.bits_compress_target(text)
		return last_lang, necessary_bits



	def locate_chunks_lang(self):
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)

		n_chunk, lang, previous_n_bits = 0, None, 0

		while True:
			target_text = f.read(self.CHUNK_SIZE)

			if target_text == '':
				return

			lang, n_bits = self.guess_language(target_text, lang)

			if previous_n_bits and n_bits / previous_n_bits - 1 >= self.THRESHOLD:
				lang, n_bits = self.guess_language(target_text, lang, ignore_lang=True)

			previous_n_bits = n_bits

			self.chunk_lang[n_chunk] = lang.ref_filename
			logging.info(f"Guessed language: {lang.ref_filename}")

			n_chunk += 1


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
		[lang.run(t_alphabet) for lang in self.langs]

		self.locate_chunks_lang()
