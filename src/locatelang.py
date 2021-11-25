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
		self.THRESHOLD = 0.05


	def get_langs(self):
		try:
			return [Lang(f"{self.dir_ref_files}{ref_file}", self.target_filename, self.k, self.alpha)\
				for ref_file in os.listdir(self.dir_ref_files)]
		except:
			logging.info(f"Directory {self.dir_ref_files} not found")
			sys.exit()



	def guess_language(self, text, last_lang):
		if not last_lang:
			necessary_bits = [lang.bits_compress_target(text) for lang in self.langs]
			min_bits = min(necessary_bits)
			return self.langs[necessary_bits.index(min_bits)], min_bits

		necessary_bits = last_lang.bits_compress_target(text)
		return last_lang, necessary_bits



	def locate_chunks_lang(self):
		try:
			f = open(self.target_filename, 'r')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)

		last_n_bits = 0
		last_lang = None
		n_chunk = 0

		while True:
			target_text = f.read(self.CHUNK_SIZE)

			if target_text == '':
				return

			lang, n_bits = self.guess_language(target_text, last_lang)
			
			if last_lang and lang.ref_filename == last_lang.ref_filename:
				if n_bits / last_n_bits <= self.THRESHOLD:
					last_lang = None
			else:
				last_lang = lang
			last_n_bits = n_bits

			logging.info(f"Guessed language: {lang.ref_filename}")
			self.chunk_lang[n_chunk] = lang.ref_filename
			n_chunk += 1


	def run(self):
		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		[lang.run() for lang in self.langs]

		self.locate_chunks_lang()
