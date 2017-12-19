# This code was inspired by http://github.com/willf/segment


import os

class Analyzer(object):
    def __init__(self,file_dir="data/dictionary/en" , case_folding=True, minimum_frequency=1.0e-08):
        self.frequencies = dict()
        self.total = 0.0
        self.minimum_frequency = minimum_frequency
        self.file_dir =file_dir
        self.total = float(open(file_dir + "/total.tsv").readlines()[0])
        self.case_folding = case_folding
        counts = dict()
        for line in open(file_dir + "/frequencies.tsv"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                word, count = parts
                if case_folding:
                    key = word.lower()
                else:
                    key = word
                counts[key] = counts.get(word, 0.0) + float(count)
            else:
                pass

        for key in counts:
            frequency = counts[key] / self.total
            if frequency >= self.minimum_frequency:
                self.frequencies[key] = frequency

    def frequency(self, word):
        return self.frequencies.get(word, 0.0)

    def segment(self, text):
        # best[i] : best log probability for text[0:i]
        # words[i] : best word ending at position i
        original_text = text
        if self.case_folding:
            text = text.lower()
        n = len(text)
        # strings can be split into unicode chars using list
        words = [''] + list(text)
        best = [1.0] + [0.0] * n
        # fill in vectors best, words via dynamic programming
        for i in range(n + 1):
            for j in range(i):
                w = text[j:i]
                lp = self.frequency(w) * best[i - len(w)]
                if lp >= best[i]:
                    best[i] = lp
                    words[i] = original_text[j:i]
        # now recover the sequence of best words
        seq = []
        i = len(words) - 1
        while i > 0:
            # prevent an infinite loop from occuring here
            if len(words[i]) > 0:
                seq.append(words[i])
                i -= len(words[i])
            else:
                i -= 1
        # reverse sequence
        return seq[::-1]