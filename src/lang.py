import logging
from fcm import FCM
from math import log2
import sys


class Lang:
	def __init__(self, reference_filename, target_filename, k=3, alpha=0.1) -> None:
		self.ref_filename = reference_filename
		self.target_filename = target_filename
		self.k = k
		self.alpha = alpha
		self.fcm = FCM(k,alpha,reference_filename)

		self.t_number_chars = 0 # number of chars in target file
		self.n_bits = 0 # number of bits to compress the text
	
	
	def train_fcm(self):
		logging.info(f"Starting to train FCM with file {self.ref_filename}")
		self.fcm.run(get_alphabet=False)

	
	def merge_alphabets(self):
		try:
			file_text = open(self.target_filename,"r")
		except FileNotFoundError:
			print(f"Could not open file {self.target_filename}")
			sys.exit(0)

		max_index = max(self.fcm.alphabet.values())

		for line in file_text:
			for char in line:
				self.t_number_chars += 1
				if char not in self.fcm.alphabet:
					max_index += 1
					self.fcm.alphabet[char] = max_index
					self.fcm.alphabet_size += 1
					logging.info(f'Adding char {char} to reference alphabet')


	def run(self):
		self.fcm.read_file()

		self.merge_alphabets()
		
		self.train_fcm()

		self.bits_compress_target()

	
	def bits_compress_target(self):
		logging.info(f"Calculating number of bits to compress {self.target_filename} with a trained FCM with the {self.ref_filename} file.")

		with open(self.target_filename, 'r') as f:
			target_text = f.read()

			context = target_text[:self.k]

			# alpha_x_alphabet_size = self.fcm.alpha * self.fcm.alphabet_size
			# total_possible_seq = self.fcm.alphabet_size ** (self.k + 1)
			# total_occ_alpha = self.fcm.total_occurrences + self.fcm.alpha * total_possible_seq

			for next_char in target_text[self.k:]:
				context_probabilities = self.fcm.get_context_probabilities(context)

				# if self.fcm.is_hash_table:
				# 	context_probability = (self.fcm.prob_table[context]["total_occur"] + alpha_x_alphabet_size)\
				# 		/ total_occ_alpha
				# else:
				# 	context_index = self.fcm.get_context_index(context)
				# 	context_probability = (self.fcm.prob_table[context_index][-1] + alpha_x_alphabet_size)\
                #     	/ total_occ_alpha

				next_char_index = self.fcm.alphabet[next_char]
				self.n_bits -= log2(context_probabilities[next_char_index])

				context = context[1:] + next_char

		logging.info(f"Finished calculating. The number of bits necessary are {self.n_bits}")

		return self.n_bits
