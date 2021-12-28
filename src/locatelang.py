import logging
from math import log
from lang import Lang
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from utils import open_file, open_dir
from math import log2

class LocateLang:
	def __init__(self, dir_ref_files, target_filename, k=3, alpha=0.01, multi_k=[], threshold_alphabet=False) -> None:
		self.dir_ref_files = dir_ref_files if dir_ref_files[-1] == '/' or dir_ref_files[-1] == "\\" else f"{dir_ref_files}/"
		self.target_filename = target_filename
		self.k = k
		if not multi_k:
			multi_k.append(k)
		self.multi_k = multi_k
		self.alpha = alpha

		self.location_langs = {}

		self.langs = self.get_langs()

		self.threshold_alphabet = threshold_alphabet

		self.CHUNK_SIZE = 10_000
		self.CHUNKS_THRESHOLD = 0.1
		self.AVERAGE_THRESHOLD = 0.40
		# TODO: adjust window size
		self.WINDOW_SIZE = 8 * max(self.multi_k)

		self.strategy = self.get_best_strategy()

	
	def get_best_strategy(self):
		# less than .2 MB
		if os.path.getsize(self.target_filename) * 10 ** -6 < .2:
			return "windows"
		return "chunks"


	def get_langs(self):
		ref_files = open_dir(self.dir_ref_files)
		langs = defaultdict(lambda: [])

		[langs[k].append(Lang(f"{self.dir_ref_files}{ref_file}", self.target_filename, k, self.alpha))
			for k in self.multi_k
			for ref_file in ref_files]

		return langs

	def guess_language(self, text, last_lang, ignore_lang=False, y_axis=None):
		if not last_lang or ignore_lang:
			lang_bits = [lang.bits_compress_target(text) if lang != last_lang else lang.n_bits for lang in self.langs[self.k]]
			min_bits = min(lang_bits)
			return self.langs[self.k][lang_bits.index(min_bits)], min_bits, lang_bits

		lang_bits = [last_lang.bits_compress_target(text)]
		return last_lang, lang_bits[0], lang_bits
	
	
	def get_langs_bits(self, text):
		langs_bits = {}
		total_bits = 0
		for lang in self.langs[self.k]:
			n_bits = lang.bits_compress_target(text)
			langs_bits[lang.lang_name] = n_bits
			total_bits += n_bits
		
		return langs_bits, total_bits


	def locate_windows_lang(self):
		logging.info("Starting Locating Langs for each window")
		f = open_file(self.target_filename, 'r')

		window_langs, target_text, lang_y, x_pos, lang_id, thresholds =\
			defaultdict(lambda: []), f.read(), defaultdict(lambda: []), [], {}, {}

		f.close()
		
		for k in self.multi_k:
			for i, lang in enumerate(self.langs[k]):
				lang_id[lang.lang_name] = i
			break

		for lang_name, _id in lang_id.items():
			sum_entropies = 0
			for k in self.multi_k:
				sum_entropies += self.langs[k][_id].fcm.entropy

			if not self.threshold_alphabet:
				thresholds[lang_name] = sum_entropies / len(self.multi_k)
			else:
				thresholds[lang_name] = (sum_entropies / len(self.multi_k) + log2(self.langs[k][_id].fcm.alphabet_size) / 2) / 2


		previous_pos = (0, 0)
		for initial_pos in range(len(target_text)-self.WINDOW_SIZE):
			end_pos = initial_pos + self.WINDOW_SIZE
			window_text = target_text[initial_pos:end_pos]
			x_pos.append(end_pos)

			for lang_name, _id in lang_id.items():
				previous_window_lang = lang_name in window_langs[previous_pos]
				window_bits = 0
				for k in self.multi_k:
					window_bits += self.langs[k][_id].bits_compress_target(window_text)
				average_window_bits = window_bits / (self.WINDOW_SIZE * len(self.multi_k))

				lang_y[lang_name].append(average_window_bits)
				# noise reduction
				# if the previous window was of this language then increase 5 % of the threshold 
				if not previous_window_lang and average_window_bits <= thresholds[lang_name] or\
					previous_window_lang and average_window_bits <= thresholds[lang_name] * 1.05:
					window_langs[(initial_pos, end_pos)].append(lang_name)

			previous_pos = (initial_pos, end_pos)
		return window_langs, lang_y, x_pos, thresholds


	def locate_chunks_lang(self):
		f = open_file(self.target_filename, 'r')

		location_langs, n_chunk, lang, previous_n_bits, lang_y, x_pos, end_pos =\
			defaultdict(lambda: []), 0, None, 0, defaultdict(lambda: []), [], 0

		while True:
			target_text = f.read(self.CHUNK_SIZE)

			if target_text == '':
				return location_langs, lang_y, x_pos

			len_target_text = len(target_text)

			start_pos = end_pos + 1
			end_pos = start_pos + len_target_text
			pos = (start_pos, end_pos)
			x_pos.append(end_pos)
			lang, n_bits, lang_bits = self.guess_language(target_text, lang)

			if previous_n_bits and n_bits / previous_n_bits - 1 >= self.CHUNKS_THRESHOLD:
				lang, n_bits, lang_bits = self.guess_language(target_text, lang, ignore_lang=True)
				[lang_y[self.langs[self.k][i].lang_name].append(bits) for i, bits in enumerate(lang_bits)]
			else:
				lang_y[lang.lang_name].append(n_bits)
				[lang_y[other_lang.lang_name].append(-50000) for other_lang in self.langs[self.k] if other_lang.lang_name != lang.lang_name]

			previous_n_bits = n_bits
			location_langs[pos].append(lang.lang_name)

			n_chunk += 1


	def compare_lang(self):
		logging.info("Starting Locating Langs for each window")
		
		f = open_file(self.target_filename, 'r')
		
		location_langs, window_langs, total_bits, target_text =\
			defaultdict(lambda: []), {}, 0, f.read()

		f.close()

		window = target_text[:self.WINDOW_SIZE]

		for ind, next_char in enumerate(target_text[self.WINDOW_SIZE:]):
			langs_bits, w_total_bits = self.get_langs_bits(window)
			total_bits += w_total_bits
			window_langs[(ind, ind + self.WINDOW_SIZE)] = langs_bits

			window = window[1:] + next_char

		average_bits = total_bits / ind / len(self.langs[self.k])

		logging.info(f"Average number of bits for each window: {average_bits}")

		x_pos = [end_pos for _, end_pos in window_langs.keys()]
		lang_y = defaultdict(lambda: [])

		# at least 40 % smaller than average
		threshold = average_bits * self.AVERAGE_THRESHOLD

		for w, langs in window_langs.items():
			for lang, n_bits in langs.items():
				lang_y[lang].append(n_bits)
				if n_bits <= threshold:
					location_langs[w].append(lang)

		return location_langs, lang_y, x_pos, average_bits


	def merge_locations(self, location_langs):
		previous_langs, previous_start_pos, final_location_langs =\
			[], 0, {}

		sorted_locations_langs = sorted(location_langs.items(), key=lambda k: k[0])

		for loc, langs in sorted_locations_langs:
			start_pos, end_pos = loc

			if Counter(previous_langs) == Counter(langs):
				continue

			if start_pos != 0:
				final_location_langs[(previous_start_pos, start_pos - 1)] = previous_langs

			previous_start_pos = start_pos
			previous_langs = langs

		# last location
		if (previous_start_pos, end_pos) not in final_location_langs:
			final_location_langs[(previous_start_pos, end_pos)] = previous_langs
		return final_location_langs


	def plot_results(self, x_pos=None, lang_y=None, average_bits=None, thresholds=None, final_location_langs={}):
		colors = list(mcolors.BASE_COLORS) + list(mcolors.CSS4_COLORS.values())
		label_colors = {lang.lang_name: colors[i] for i, lang in enumerate(self.langs[self.k])}

		if final_location_langs:
			label_langs = {lang.lang_name: i for i, lang in enumerate(self.langs[self.k])}
			patches = [mpatches.Patch(color=color, label=lang_name) for lang_name, color in label_colors.items()]

			for loc, langs in final_location_langs.items():
				for lang in langs:
					plt.plot(loc, [label_langs[lang]] * 2, color=label_colors[lang])
		
			plt.yticks(list(label_langs.values()), list(label_langs.keys()))

			plt.xlabel("Characters")

			plt.legend(handles=patches)
		else:
			for lang, y in lang_y.items():
				plt.plot(x_pos, y, 'o', label=f"{lang} Average Bits", color=label_colors[lang])
			
			plt.ylim(bottom=0)
			plt.ylabel("Number of Bits")
			plt.xlabel("Characters")

			if average_bits:
				plt.plot(x_pos, [average_bits] * len(x_pos), label='Total Average Bits')
				plt.plot(x_pos, [average_bits * self.AVERAGE_THRESHOLD] * len(x_pos), label='Threshold')
				plt.ylim(0, average_bits * 1.1)
			
			if thresholds:
				[plt.plot(x_pos, [threshold] * len(x_pos), label=f"{lang} Threshold", color=label_colors[lang])\
					for lang, threshold in thresholds.items()]
				plt.ylim(0, max(thresholds.values()) * 2)
			
			plt.legend()
		plt.show()


	def get_t_alphabet(self):
		file_text = open_file(self.target_filename, "r")

		t_alphabet = set()

		for line in file_text:
			for char in line:
				t_alphabet.add(char)

		file_text.close()

		return t_alphabet


	def run(self, compare_langs=False):
		t_alphabet = self.get_t_alphabet()

		logging.info(f"Starting to train FCM with files inside {self.dir_ref_files}")
		[lang.run(t_alphabet) for k in self.multi_k for lang in self.langs[k]]

		if self.strategy == "chunks":
			location_langs, lang_y, x_pos = self.locate_chunks_lang()
			self.plot_results(x_pos=x_pos, lang_y=lang_y)
		elif compare_langs:
			location_langs, lang_y, x_pos, average_bits = self.compare_lang()
			self.plot_results(x_pos=x_pos, lang_y=lang_y, average_bits=average_bits)
		else:
			location_langs, lang_y, x_pos, thresholds = self.locate_windows_lang()
			self.plot_results(x_pos=x_pos, lang_y=lang_y, thresholds=thresholds)

		self.location_langs = self.merge_locations(location_langs)
		self.plot_results(final_location_langs=self.location_langs)
