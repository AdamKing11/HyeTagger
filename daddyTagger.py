import re, sys, nltk
from random import shuffle
from bclass_cross import *

"""
	one tagger to rule them all!

	take our morphological tagger (baby) and our syntactic 
	tagger (momma) and combine them into a single tagger

	To tag, we go through a sentence and:
		if it's in our list of UNambiguous tags, just tag

		if it's in our list of AMBIGUOUS tags, we get the possible
			tags and let contextual probability be the tie-breaker


		if it's unknown, we run it through our Bayesian classifier
		(baby) and ALSO use contextual probs
	"""

from word_features import *
from babyTagger import *
from mommaTagger import *

class daddyTagger:
	
	babyTagger = 0
	mommaTagger = 0

	def __init__(self, bt, mt):
		"""
		"""
		self.babyTagger = bt
		self.mommaTagger = mt


if __name__ == "__main__":
	b = c_load("b_tagger.t")
	m = c_load("m_tagger.t")

	daddy = daddyTagger(b, m)
	#for u in daddy.babyTagger.Unambig:
	#	print(u)
	print("V\t\tN")
	print(daddy.mommaTagger.tag_trigrams[("A", "V", "A")], daddy.mommaTagger.tag_trigrams[("A", "N", "A")])
	print(daddy.mommaTagger.quick_tag(("N","V"),"A","A"))
	