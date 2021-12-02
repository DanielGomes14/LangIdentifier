import logging
from math import log
from fcm import FCM
from lang import Lang
import os
import sys
from collections import defaultdict, Counter


class LocateLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.1) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.k = k
		self.alpha = alpha

		self.langs = self.get_langs()

		self.location_langs = {}

		self.CHUNK_SIZE = 10_000
		self.CHUNKS_THRESHOLD = 0.02
		self.WINDOW_THRESHOLD = 0.25

		self.strategy = self.get_best_strategy()

	
	def get_best_strategy(self):
		# less than .2 MB
		if os.path.getsize(self.target_filename) * 10 ** -6 < .2:
			return "windows"
		return "chunks"


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
	
	
	def get_langs_bits(self, text):
		langs_bits = {}
		total_bits = 0
		for lang in self.langs:
			n_bits = lang.bits_compress_target(text)
			langs_bits[lang.ref_filename] = n_bits
			total_bits += n_bits
		
		return langs_bits, total_bits
	

	def locate_windows_lang(self):
		logging.info("Starting Locating Langs for each window")
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)
		
		# TODO: adjust window size
		location_langs, window_size, window_langs, total_bits, target_text =\
			defaultdict(lambda: []), 5, {}, 0, f.read()

		window = target_text[:window_size]

		for ind, next_char in enumerate(target_text[window_size:]):
			langs_bits, w_total_bits = self.get_langs_bits(window)
			total_bits += w_total_bits
			window_langs[(ind, ind + window_size)] = langs_bits

			window = window[1:] + next_char

		average_bits = total_bits / ind / len(self.langs)
		
		logging.info(f"Average number of bits for each window: {average_bits}")
		
		for w, langs in window_langs.items():
			for lang, n_bits in langs.items():
				# 25 % smaller than average
				if (average_bits - n_bits) / average_bits >= self.WINDOW_THRESHOLD:
					location_langs[w].append(lang)

		return location_langs


	def locate_chunks_lang(self):
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)

		location_langs, n_chunk, lang, previous_n_bits =\
			defaultdict(lambda: []), 0, None, 0

		while True:
			target_text = f.read(self.CHUNK_SIZE)

			if target_text == '':
				return location_langs

			lang, n_bits = self.guess_language(target_text, lang)

			if previous_n_bits and n_bits / previous_n_bits - 1 >= self.CHUNKS_THRESHOLD:
				lang, n_bits = self.guess_language(target_text, lang, ignore_lang=True)

			previous_n_bits = n_bits
			start_pos = n_chunk * self.CHUNK_SIZE
			location_langs[(start_pos, start_pos + self.CHUNK_SIZE)].append(lang.ref_filename)
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


	def merge_locations(self, location_langs):
		previous_langs, previous_start_pos, previous_end_pos = [], 0, 0

		for loc, langs in location_langs.items():
			start_pos, end_pos = loc
			if Counter(previous_langs) == Counter(langs):
				previous_end_pos = end_pos
			else:
				if previous_end_pos:
					self.location_langs[(previous_start_pos, previous_end_pos)] = previous_langs
				previous_start_pos, previous_end_pos = start_pos, end_pos
			previous_langs = langs

		# last location
		if (previous_start_pos, previous_end_pos) not in self.location_langs:
			self.location_langs[(previous_start_pos, previous_end_pos)] = previous_langs
		

	def run(self):
		t_alphabet = self.get_t_alphabet()

		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		[lang.run(t_alphabet) for lang in self.langs]

		if self.strategy == "chunks":
			self.merge_locations(self.locate_chunks_lang())
		else:
			self.merge_locations(self.locate_windows_lang())
