import re, sys, nltk
from random import shuffle

from lib.word_features import *
from lib.bclass_cross import *

class mommaTagger:

	accuray = 1.

	tag_trigrams = {} # we hold in counts for each tag trigram

	all_tags = set([])

	mid_prob = {}	# a dict for holding the probabilities of a given tag
					# given the tags to the left and right
					# key formet = ((PREV, FOLLOWING), TAG)


	def __init__(self, tagged_c, verbose = True):
		"""
		trains a trigram probability classifier
		we load in the tagged corpus from the babyTagger file
		and use that to calculate trigram probabilites for tags
		"""
		self.tag_trigrams, self.all_tags = self.read_tag_trigrams(tagged_c)
		self.mid_prob = self.build_mp_dict()
		
	def read_tag_trigrams(self, baby_tagged_file, threshold = 1., verbose = True):
		"""
		read in a file tagged by the baby tagger and get all tag
		trigrams where ALL tokens in the trigram beat our threshold
		(default 1, so we only do unambiguous tokens)
		"""
		all_tags = set([])
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
					# and we'll add it to the list of all tags...
					all_tags.add(l[2])
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
				len(trigrams), "trigram types.")
		return trigrams, all_tags


	def prob_last(self, context, target):
		"""
		given the context (previous two tags), what's the probability of the target tag?
		i.e. given (A, V), what's the probability that we have an N?
		which is p(N|A,V) / sum(p(TAG|A,V)) for all TAGs

		'context' as tuple
		'target' as single tag

		"""
		try:
			return self.tag_trigrams[(context[0], context[1], target)] / sum([self.tag_trigrams[c] \
				for c in self.tag_trigrams if c[0] == context[0] and c[1] == context[1]])
		except:
			return 0.

	def prob_middle(self, context, target):
		""" 
		probability of a tag given it's previous and following tag
		"""
		try:
			return self.tag_trigrams[(context[0], target, context[1])] / sum([self.tag_trigrams[c] \
				for c in self.tag_trigrams if c[0] == context[0] and c[2] == context[1]])
		except:
			return 0.

	def build_mp_dict(self):
		"""
		just calculate the probabilities of a TAG given what comes before and after once at the beginning
		then we save it in a dictionary so it's faster!
		"""
		pd = {}
		pos_tri = [(x, y, z) for x in self.all_tags for y in self.all_tags for z in self.all_tags]
		for p in pos_tri:
			pd[p] = self.prob_middle((p[0], p[2]), p[1])
		return pd

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
	
	#c_save(momma, "taggers/m_tagger.t")
	momma = c_load("taggers/m_tagger.t")

	
	print("V\t\tN")
	print(momma.tag_trigrams[("A", "V", "A")], momma.tag_trigrams[("A", "N", "A")])
	print(momma.mid_prob[("A", "V", "A")], momma.mid_prob[("A", "N", "A")])
	print(momma.mid_prob[("N", "N", "N")])
	print(momma.quick_tag(("N","V"),"A","A"))
	
	"""
	print(momma.tag_trigrams[("N", "V", "A")], momma.tag_trigrams[("N", "N", "A")])
	print(momma.quick_tag(("N","V"),"N","A"))
	
	print(momma.tag_trigrams[("N", "V", "N")], momma.tag_trigrams[("N", "N", "N")])
	print(momma.quick_tag(("N","V"),"N","N"))
	
	print(momma.tag_trigrams[("V", "V", "V")], momma.tag_trigrams[("V", "N", "V")])
	print(momma.quick_tag(("N","V"),"V","V"))

	print(momma.tag_trigrams[("N", "V", "V")])
	print(momma.tag_trigrams[("N", "V", "V")] / sum([momma.tag_trigrams[c] for c in momma.tag_trigrams if c[0] =="N" and c[1] == "V"]))
	print(momma.prob_last(("N","V"),"V"))
	"""