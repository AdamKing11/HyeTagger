import re, sys, nltk
import numpy as np
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

	def tag(self, s, threshold = .9):
		"""
			Tag a sentence! (duh)
			We first go through and find all unambiguous words in the sentence
			then, we use those unambiguous words to give us syntactic context 
			(i.e. tag trigams) information to help break ties for ambiguous tokens
			then we do the same for totally novel words
		"""
		# build an empty list of length s
		tagged = ["" for _ in s]
	
		# first pass, find and tag all unambiguous tokens or reeeeallly high prob
		# unknown words
		for i in range(len(s)):
			w = s[i]
			g = self.babyTagger.quick_tag(w)
				# if it's unambiguous....
			if g[0] == 1:
				tagged[i] = g[2][1]
				# if it's an unknown word AND it beats our threshold for close enough
				# to unambiguous...
			elif g[0] == 3 and g[2][0] > threshold:
				tagged[i] = g[2][1]

		# second pass, fill in any gaps
		# how we do this is we find the last good tag before a gap and get a substring
		# until the next good tag
		# then we pass that substring to a function that finds the optimal set of tags
		for i in range(len(s)):
			if s[i] != "":
				continue

		print(tagged)

	def gap_fill(self, s, threshold = .9):
		"""
		we're going to find the best fit of tags for a given substring (by considering all)
		possible

		because we do this in small chunks, we reduce that HUGE factorial amount of possiblities
		we'd need to consider for the whole string....

		okay, this is going to be fun. Because we have a length s-2 string, with up to 3 possible
		tags for each word AND up to 9 possible tag trigrams, we are going  to be building a 3D
		matrix of size (s-2) * 3 * 9....
		and doing 3d viterbi
		""" 
		# get the first and last tag of the sentence part
		first_tag = self.babyTagger.quick_tag(s[0])[2][1]
		last_tag = self.babyTagger.quick_tag(s[-1])[2][1]
		
		# need to calculate exactly how many possible sentences we'll be considering...
		
		tag_matrix = [[first_tag]]
		# first, make matrix of all possible tags....
		for i in range(len(s)-2):
			w = s[i+1]
			pos_tags = self.babyTagger.quick_tag(w)[2]
			tags = [pos_tags[k] for k in np.arange(1,len(pos_tags),2)]
			tag_matrix.append(tags)
		tag_matrix.append([last_tag])
		
		prob_matrix = np.ones((len(s)-2,3,3))
		# loop through all unknown tags (we know the edges are good so len-2)
		for i in range(len(s)-2):
			w = s[i+1]
			pos_tags = self.babyTagger.quick_tag(w)[2]
			probs = [pos_tags[k] for k in np.arange(0,len(pos_tags),2)]
			tags = [pos_tags[k] for k in np.arange(1,len(pos_tags),2)]
			# from 0 to end of pos_tags, step by 2 (because we want to skip the actual tags)
			for j in range(len(probs)):
				for k in range(len(tag_matrix[j])):
					print(tag_matrix[j][k])
				prob_matrix[i,j,0] *= probs[j]
				
			
		print(prob_matrix)




if __name__ == "__main__":
	b = c_load("b_tagger.t")
	m = c_load("m_tagger.t")

	daddy = daddyTagger(b, m)
	#for u in daddy.babyTagger.Unambig:
	#	print(u)
	#print("V\t\tN")
	#print(daddy.mommaTagger.tag_trigrams[("A", "V", "A")], daddy.mommaTagger.tag_trigrams[("A", "N", "A")])
	#print(daddy.mommaTagger.quick_tag(("N","V"),"A","A"))
	s = "<s> Դուք պետք է հավաստեք , որ ձեր ներլցած ֆայլը ոչ մի հեղինակային իրավունք չի խախտում ։ </s>"
	s = s.rsplit(" ")
	#daddy.tag(s)
	z = "<s> Դուք Դուք պետք"
	z = z.split(" ")
	daddy.gap_fill(z)
	
	