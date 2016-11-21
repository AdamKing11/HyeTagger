import re, sys, nltk
from random import shuffle

from lib.word_features import *
from lib.bclass_cross import *
from lib.test_eval import *

class mommaTagger:

	tag_trigrams = {} # we hold in counts for each tag trigram

	total_trigrams = 0	# for ALL trigrams we've seen
	all_tags = set([])

	mid_prob = {}	# a dict for holding the probabilities of a given tag
					# given the tags to the left and right
					# key formet = ((PREV, FOLLOWING), TAG)


	def __init__(self, tagged_c, tagged_c_type = 1, verbose = True):
		"""
		trains a trigram probability classifier
		we load in the tagged corpus from the babyTagger file
		and use that to calculate trigram probabilites for tags
		"""
		# type 1 for each word being on its own line
		if tagged_c_type == 1:
			self.tag_trigrams, self.all_tags = self.read_tag_trigrams(tagged_c)
		if tagged_c_type == 2:
			self.tag_trigrams, self.all_tags = self.read_hand_tagged(tagged_c)
		self.mid_prob = self.build_mp_dict()
		self.total_trigrams = self.count_all_trigrams()
	
	def forget(self, trigrams_to_f, verbose = True):
		"""
		take in a dictionary of trigrams and "forget" seeing them, ie remove them
		from our counts of trigrams
		"""
		total_forgotten = 0
		
		if verbose:
			print("Starting with", self.total_trigrams, "total trigrams.")
			print("Now **forgetting** trigrams across", len(trigrams_to_f), "trigram types.")
		for t in trigrams_to_f:
			if t in self.tag_trigrams:
				self.tag_trigrams[t] -= trigrams_to_f[t]
				total_forgotten += trigrams_to_f[t]

		if verbose:
			print("\t",total_forgotten, "total forgotten.")
			print("Using", len(self.tag_trigrams), "to calculate probabilities....")
		self.mid_prob = self.build_mp_dict()	
		self.total_trigrams = self.count_all_trigrams()
		if verbose:
			print("Done. We now have", self.total_trigrams, "across", \
				len(self.tag_trigrams), "types.")

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

	def read_hand_tagged(self, hand_tagged_file, verbose = True):
		"""
		reads a tagged corpus where each sentence is on its own line AND word/tags are
		in the format: w_t w_t w_t" etc
		"""
		all_tags = set([])
		trigrams = {}
		if verbose:
			print("Opening the hand tagged file", hand_tagged_file, "now.")
		with open(hand_tagged_file, "r") as rf:
			i, j = 0, 0
			for l in rf:
				i += 1
				l = l.rstrip().rsplit("\t")
				for i in range(2, len(l)):
					j += 1
					# get the tag 2 ago, 1 ago and current
					# probably a better way to do this, rather than re-calc every time....
					laster_tag = split_tagged_lemma(l[i-2])[1]
					all_tags.add(laster_tag)
					
					last_tag = split_tagged_lemma(l[i-1])[1]
					all_tags.add(last_tag)
					
					tag = split_tagged_lemma(l[i])[1]
					all_tags.add(tag)

					tg = (laster_tag, last_tag, tag)
					
					if tg in trigrams:
						trigrams[tg] += 1
					else:
						trigrams[tg] = 1
		if verbose:
			print("Read in", i, "sentences and found", j, "trigrams with", len(all_tags), \
				"total trigram types.")
		return trigrams, all_tags

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

	def count_all_trigrams(self):
		"""
		goes through all trigram types and finds how many of each we have and returns sum
		"""
		tc = 0
		for t in self.tag_trigrams:
			tc += self.tag_trigrams[t]
		return tc


if __name__ == "__main__":
	#momma = mommaTagger("tagged.wiki.txt")
	
	momma = mommaTagger("EANC.200.hand.txt", tagged_c_type = 2)
	#c_save(momma, "taggers/m_tagger.t")
	#momma = c_load("taggers/m_tagger.t")
	
	#train, test = split_corpus("EANC.golds.txt")
	#trigrams_to_f = count_trigrams(test)

	#momma.forget(trigrams_to_f)


	
	"""
	print("V\t\tN")
	print(momma.tag_trigrams[("A", "V", "A")], momma.tag_trigrams[("A", "N", "A")])
	print(momma.mid_prob[("A", "V", "A")], momma.mid_prob[("A", "N", "A")])
	print(momma.mid_prob[("N", "N", "N")])
	print(momma.quick_tag(("N","V"),"A","A"))
	
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