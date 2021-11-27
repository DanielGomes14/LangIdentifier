from collections import defaultdict
from math import log2
import sys
FILENAME = "./../example/example.txt"

class FCM:
    def __init__(self, k=3, alpha=0.1, filename=FILENAME) -> None:
        assert alpha > 0, "Values of alpha must be greater than zero"
        assert k > 0, "Values of k must be greater than zero"

        self.k = k
        self.alpha = alpha
        self.filename = filename

        self.alphabet = {}
        self.alphabet_size = 0

        self.prob_table = []
        self.is_hash_table = False

        self.total_occurrences = 0
        
        self.entropy = 0



    def run(self, get_alphabet=True):
        if get_alphabet:
            # get alphabet and store characters' indexes
            self.read_file()
    
        # initialize table that stores occurencies
        self.setup_table()

        # set occurrences on the table
        self.set_occurrences()

        # replace the occurrences with the calculated probabilities
        self.calc_probabilities()

        # calculate entropy
        self.calc_entropy()


    def read_file(self):
        try:
            file_text = open(self.filename, "r", encoding='utf-8')
        except FileNotFoundError:
            print(f"Could not Open file {self.filename}")
            sys.exit(0)

        # get alphet and store indexes
        ind_counter = 0
        for line in file_text:
            for char in line:
                # non-aplhabetical count
                # Lower upper case
                if char not in self.alphabet:
                    self.alphabet[char] = ind_counter
                    ind_counter += 1

        # store size of alphabet
        self.alphabet_size = ind_counter


    def setup_table(self):
        # memory formula
        if self.in_memory_limit():
            self.prob_table = [[0] * (self.alphabet_size+1) for _ in range(self.alphabet_size ** self.k)]
        else:
            self.prob_table = defaultdict(lambda: defaultdict(int))
            self.is_hash_table = True


    def set_occurrences(self):
        context = ""

        file_text = open(self.filename, "r", encoding='utf-8')
        for line in file_text:
            for char in line:
                # We should only use sequences of size k
                if len(context) == (self.k):
                    # add to occurrences table one more occurency of this char for the given context
                    self.add_occur_to_table(context, char)
                    # clear first char of context and add next_char
                    context = context[1:] + char
                    continue
                context += char


    def in_memory_limit(self) -> bool:
        # verifies situations in which Memory Usage will be higher than 0.5 MB
        mem = (self.alphabet_size ** self.k ) * self.alphabet_size * 16/8/1024/1024
        return mem <= 0.5


    def get_context_index(self, context):
        context_index = 0

        for i in range(self.k, 0, -1):
            context_index += self.alphabet[context[self.k-i]] * self.alphabet_size**(i-1)
        
        return context_index


    def add_occur_to_table(self, context, next_char):
        self.total_occurrences += 1

        if self.is_hash_table:
            self.prob_table[context][next_char] += 1
            self.prob_table[context]["total_occur"] += 1
            return

        context_index = self.get_context_index(context)
        self.prob_table[context_index][self.alphabet[next_char]] += 1
        self.prob_table[context_index][-1] += 1


    def calc_probabilities(self) -> None:
        if self.is_hash_table:
            for context, next_occur_chars in self.prob_table.items():
                total_row_occur = self.prob_table[context]["total_occur"]

                divisor = total_row_occur + self.alpha * self.alphabet_size
                
                for next_char, num_occur in next_occur_chars.items():
                    if next_char != "total_occur":
                        self.prob_table[context][next_char] = (num_occur + self.alpha) / divisor
            return

        for context_row in self.prob_table:
            total_row_occur = context_row[-1]

            divisor = total_row_occur + self.alpha * self.alphabet_size

            for i in range(self.alphabet_size):
                context_row[i] = (context_row[i] + self.alpha) / divisor


    def get_context_probabilities(self, context):
        if self.is_hash_table:
            final_probs = [0] * self.alphabet_size
            total_row_occur = self.prob_table[context]["total_occur"]
            divisor = total_row_occur + self.alpha * self.alphabet_size
            for symbol, index in self.alphabet.items():
                prob = self.prob_table[context][symbol]
                if prob:
                    final_probs[index] = prob
                else:
                    final_probs[index] = self.alpha / divisor

            return final_probs
        else:
            context_index = self.get_context_index(context)
            return self.prob_table[context_index][:-1]


    def calc_entropy(self):
        alpha_x_alphabet_size = self.alpha * self.alphabet_size
        total_possible_seq = self.alphabet_size ** (self.k + 1)
        total_occ_alpha = self.total_occurrences + self.alpha * total_possible_seq

        if self.is_hash_table:
            num_non_occ_seq = self.alphabet_size ** self.k - len(self.prob_table)

            for context in self.prob_table.keys():
                context_entropy = 0

                context_probability = (self.prob_table[context]["total_occur"] + alpha_x_alphabet_size)\
                    / total_occ_alpha

                for prob_char in self.get_context_probabilities(context):
                    context_entropy -= prob_char * log2(prob_char)

                self.entropy += context_entropy * context_probability    

            if num_non_occ_seq:
                prob_char = self.alpha / alpha_x_alphabet_size
                context_entropy = -prob_char * log2(prob_char) * self.alphabet_size
                context_probability = alpha_x_alphabet_size / total_occ_alpha
                self.entropy += context_entropy * context_probability * num_non_occ_seq
            return
        for context_row in self.prob_table:
            context_entropy = 0

            for i in range(self.alphabet_size):
                context_entropy -= context_row[i] * log2(context_row[i])

            context_probability = (context_row[-1] + alpha_x_alphabet_size)\
                    / total_occ_alpha

            self.entropy += context_entropy * context_probability
