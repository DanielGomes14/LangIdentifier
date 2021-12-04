import logging
from math import log
from fcm import FCM
from lang import Lang
import os
import sys
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches


class LocateLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.1) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.k = k
		self.alpha = alpha

		self.langs = self.get_langs()

		self.location_langs = {}

		self.CHUNK_SIZE = 10_000
		self.CHUNKS_THRESHOLD = 0.1
		self.AVERAGE_THRESHOLD = 0.60
		# TODO: adjust window size
		self.WINDOW_SIZE = 5 * self.k

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


	def guess_language(self, text, last_lang, ignore_lang=False, y_axis=None):
		if not last_lang or ignore_lang:
			lang_bits = [lang.bits_compress_target(text) if lang != last_lang else lang.n_bits for lang in self.langs]
			min_bits = min(lang_bits)
			return self.langs[lang_bits.index(min_bits)], min_bits, lang_bits

		lang_bits = [last_lang.bits_compress_target(text)]
		return last_lang, lang_bits[0], lang_bits
	
	
	def get_langs_bits(self, text):
		langs_bits = {}
		total_bits = 0
		for lang in self.langs:
			n_bits = lang.bits_compress_target(text)
			langs_bits[lang.lang_name] = n_bits
			total_bits += n_bits
		
		return langs_bits, total_bits


	def get_lang_below_threshold(self, text, thresholds, lang_y):
		langs = []
		for lang in self.langs:
			average_bits = lang.bits_compress_target(text, calc_average=True)
			lang_y[lang.lang_name].append(average_bits)
			if average_bits <= thresholds[lang.lang_name]:
				langs.append(lang.lang_name)
		return langs


	def locate_windows_lang(self):
		#TODO: Change this to a function
		logging.info("Starting Locating Langs for each window")
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)

		window_langs, target_text, lang_y =\
			defaultdict(lambda: []), f.read(), defaultdict(lambda: [])

		thresholds = {lang.lang_name: lang.fcm.entropy for lang in self.langs}

		window = target_text[:self.WINDOW_SIZE]

		for ind, next_char in enumerate(target_text[self.WINDOW_SIZE:]):
			window_langs[(ind, ind + self.WINDOW_SIZE)] = self.get_lang_below_threshold(window, thresholds, lang_y)

			window = window[1:] + next_char

		x_pos = [end_pos for _, end_pos in window_langs.keys()]

		return window_langs, x_pos, lang_y, thresholds


	def compare_lang_averages(self):
		logging.info("Starting Locating Langs for each window")
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)
		
		location_langs, window_langs, total_bits, target_text =\
			defaultdict(lambda: []), {}, 0, f.read()

		window = target_text[:self.WINDOW_SIZE]

		for ind, next_char in enumerate(target_text[self.WINDOW_SIZE:]):
			langs_bits, w_total_bits = self.get_langs_bits(window)
			total_bits += w_total_bits
			window_langs[(ind, ind + self.WINDOW_SIZE)] = langs_bits

			window = window[1:] + next_char

		average_bits = total_bits / ind / len(self.langs)
		
		logging.info(f"Average number of bits for each window: {average_bits}")

		x_pos = [end_pos for _, end_pos in window_langs.keys()]
		lang_y = defaultdict(lambda: [])

		# at least 60 % smaller than average
		threshold = average_bits * (1 - self.AVERAGE_THRESHOLD)
		for w, langs in window_langs.items():
			for lang, n_bits in langs.items():
				lang_y[lang].append(n_bits)
				if n_bits <= threshold:
					location_langs[w].append(lang)

		return location_langs, x_pos, lang_y, average_bits


	def locate_chunks_lang(self):
		try:
			f = open(self.target_filename, 'r', encoding='utf-8')
		except FileNotFoundError:
			logging.error(f"Could not open file {self.target_filename}")
			sys.exit(0)

		location_langs, n_chunk, lang, previous_n_bits, y_axis_lang_bits =\
			defaultdict(lambda: []), 0, None, 0, defaultdict(lambda: [])

		while True:
			target_text = f.read(self.CHUNK_SIZE)

			if target_text == '':
				return location_langs, y_axis_lang_bits
			
			start_pos = n_chunk * self.CHUNK_SIZE
			pos = (start_pos, start_pos + self.CHUNK_SIZE)
			lang, n_bits, lang_bits = self.guess_language(target_text, lang)

			if previous_n_bits and n_bits / previous_n_bits - 1 >= self.CHUNKS_THRESHOLD:
				lang, n_bits, lang_bits = self.guess_language(target_text, lang, ignore_lang=True)
				[y_axis_lang_bits[self.langs[i].lang_name].append([pos, bits]) for i, bits in enumerate(lang_bits)]
			else:
				y_axis_lang_bits[lang.lang_name].append([pos, n_bits])

			previous_n_bits = n_bits
			start_pos = n_chunk * self.CHUNK_SIZE
			location_langs[pos].append(lang.lang_name)
			logging.info(f"Guessed language: {lang.lang_name}")

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
		previous_langs, previous_start_pos, previous_end_pos, final_location_langs =\
			[], 0, 0, {}

		for loc, langs in location_langs.items():
			start_pos, end_pos = loc
			if Counter(previous_langs) == Counter(langs):
				previous_end_pos = end_pos
			else:
				if previous_end_pos:
					final_location_langs[(previous_start_pos, previous_end_pos)] = previous_langs
				previous_start_pos, previous_end_pos = start_pos, end_pos
			previous_langs = langs

		# last location
		if (previous_start_pos, previous_end_pos) not in final_location_langs:
			final_location_langs[(previous_start_pos, previous_end_pos)] = previous_langs
		return final_location_langs
	

	def plot_results(self, x_pos=None, lang_y=None, average_bits=None, thresholds=None, y_axis_lang_bits=None, final_location_langs={}):
		colors = list(mcolors.BASE_COLORS) + list(mcolors.CSS4_COLORS.values())
		label_colors = {lang.lang_name: colors[i] for i, lang in enumerate(self.langs)}
		
		if x_pos:
			for lang, y in lang_y.items():
				plt.plot(x_pos, y, 'o', label=f"{lang} Average Bits", color=label_colors[lang])

			if average_bits:
				plt.plot(x_pos, [average_bits] * len(x_pos), label='Total Average Bits')
				plt.plot(x_pos, [average_bits * (1 - self.AVERAGE_THRESHOLD)] * len(x_pos), label='Average Threshold')
				plt.ylim(0, average_bits + 1)
			elif thresholds:
				[plt.plot(x_pos, [threshold] * len(x_pos), label=f"{lang_name} Threshold", color=label_colors[lang_name])\
					for lang_name, threshold in thresholds.items()]

			plt.legend()
		else:
			label_langs = {lang.lang_name: i for i, lang in enumerate(self.langs)}
			patches = [mpatches.Patch(color=color, label=lang_name) for lang_name, color in label_colors.items()]

			if y_axis_lang_bits:
				for lang, points_bits in y_axis_lang_bits.items():
					for point_bits in points_bits:
						plt.plot(point_bits[0], [point_bits[1]] * 2, color=label_colors[lang])
			else:
				for loc, langs in final_location_langs.items():
					for lang in langs:
						plt.plot(loc, [label_langs[lang]] * 2, color=label_colors[lang])
			
				plt.yticks(list(label_langs.values()), list(label_langs.keys()))

			plt.legend(handles=patches)
		plt.show()


	def run(self, compare_langs=False):
		t_alphabet = self.get_t_alphabet()

		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		[lang.run(t_alphabet) for lang in self.langs]

		if self.strategy == "chunks":
			location_langs, y_axis_lang_bits = self.locate_chunks_lang()
			self.plot_results(y_axis_lang_bits=y_axis_lang_bits)
		else:
			location_langs, x_pos, lang_y, thresholds = self.locate_windows_lang()
			self.plot_results(x_pos=x_pos, lang_y=lang_y, thresholds=thresholds)

		if compare_langs:
			avg_location_langs, x_pos, lang_y, average_bits = self.compare_lang_averages()
			self.plot_results(x_pos=x_pos, lang_y=lang_y, average_bits=average_bits)
			self.plot_results(final_location_langs=self.merge_locations(avg_location_langs))

		self.plot_results(final_location_langs=self.merge_locations(location_langs))
