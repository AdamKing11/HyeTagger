import re, sys, nltk
from random import shuffle
from bclass_cross import *

from word_features import *

class mommaTagger:

	accuray = 1.

	tag_trigrams = {} # we hold in counts for each tag trigram

	def __init__(self, tagged_c, verbose = True):
		"""
		trains a trigram probability classifier
		we load in the tagged corpus from the babyTagger file
		and use that to calculate trigram probabilites for tags
		"""
		self.tag_trigrams = self.read_tag_trigrams(tagged_c)
		
	def read_tag_trigrams(self, baby_tagged_file, threshold = 1., verbose = True):
		"""
		read in a file tagged by the baby tagger and get all tag
		trigrams where ALL tokens in the trigram beat our threshold
		(default 1, so we only do unambiguous tokens)
		"""
		trigrams = {}
		i, j = 1, 0
		with open(baby_tagged_file, "r") as rF:
			# go through lines
			tag_history = []
			for line in rF:
				line = line.rstrip()
				# split the line, l[2] is best tag, l[1] is best score
				l = line.rsplit("\t")
				# the way the tagged file is formatted has a blank line
				# between sentences and a "score card" at the end
				# we want to skip in case we have a blank line OR
				# we are at a line with < 3 fields (our score card)
				if len(l) < 3:
					if tag_history == []:
						# for counting sentences read, only want to count
						# once per sentence... we'd be counting twice since
						# there are 2 un-read lines between sentences
						i += 1
					tag_history = []
					continue
				if verbose:
					print("Reading sentence", i, end="\r")
				# if this tokens score beats our threshold, put it on the list
				if float(l[1]) >= threshold:
					tag_history.append((l[2], float(l[1])))
				else:
					# however, if the score is too low, we re-start the list 
					# and continue
					tag_history = []
					continue
				if len(tag_history) == 3:
					j += 1
					tag_trigram = tuple([t[0] for t in tag_history])
					if tag_trigram in trigrams:
						trigrams[tag_trigram] += 1
					else:
						trigrams[tag_trigram] = 1
					# now, we pop the OLDEST guy off the list
					tag_history.pop(0)
		if verbose:
			print("\nRead in", i-1, "sentences and found", j, "trigrams with", \
				len(self.tag_trigrams), "trigram types.")
		return trigrams

	def quick_tag(self, possible_tags, previous, following, trigram_weight=3):
		"""
		given a tag of the previous word and the tag of the following word, based
		on what we've seen, we return the most likely tag for target word

		MAYBE implement EM in future versions?
		"""
		tag_probabilities = {}
		for p in possible_tags:
			# probability of current tag given previous
			# which is the SUM of all trigrams where the FIRST TWO are the same,
			# divided by all trigrams where the FIRST one is the same
			forward = sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[0] == previous and c[1] == p]) / sum([self.tag_trigrams[c] \
				for c in self.tag_trigrams if c[0] == previous])
			# probability of current tag given the following tag
			# SUM of all trigrams where LAST TWO are same, divided by sum where
			# LAST one is same
			backward = sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[2] == following and c[1] == p]) / sum([self.tag_trigrams[c] \
				for c in self.tag_trigrams if c[2] == following])

			whole_context = self.tag_trigrams[(previous,p,following)] / \
				sum([self.tag_trigrams[c] for c in 	self.tag_trigrams if \
				c[0] == previous and c[2] == following])

			# now we take all 3 calculated probabilites and normalize them
			# we also weight the trigram probability HIGHER than the bigrams
			combined_prob = (forward + backward + \
				(whole_context * trigram_weight)) / (2 + trigram_weight)
			tag_probabilities[p] = combined_prob

		return tag_probabilities



if __name__ == "__main__":
	#momma = mommaTagger("tagged.txt")
	#momma.test_momma_classifier(6)
	#momma.mommaTagger.show_most_informative_features()
	
	#c_save(momma, "m_tagger.t")
	momma = c_load("m_tagger.t")
	print("V\t\tN")
	print(momma.tag_trigrams[("A", "V", "A")], momma.tag_trigrams[("A", "N", "A")])
	print(momma.quick_tag(("N","V"),"A","A"))
	
	print(momma.tag_trigrams[("N", "V", "A")], momma.tag_trigrams[("N", "N", "A")])
	print(momma.quick_tag(("N","V"),"N","A"))
	
	print(momma.tag_trigrams[("N", "V", "N")], momma.tag_trigrams[("N", "N", "N")])
	print(momma.quick_tag(("N","V"),"N","N"))
	
	print(momma.tag_trigrams[("V", "V", "V")], momma.tag_trigrams[("V", "N", "V")])
	print(momma.quick_tag(("N","V"),"V","V"))