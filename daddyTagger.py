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
		all right! going to use a quasi-Forward-Backward algorithm to find the best sequence
		of tags GIVEN a definite, known tag at the beginning and at the end
		That is, we know T_1 and T_4, so we need to find the best T_2 and T_3 to make a good
		sequence

		We're going to use the trigram probabilities of tag sequences from our unambiguously 
		tagged sentences
		""" 
	
		# get the first and last tag of the sentence part
		first_tag = self.babyTagger.quick_tag(s[0])[2][1]
		last_tag = self.babyTagger.quick_tag(s[-1])[2][1]
		
		word_tag_dict = {}	# probability of a TAG given a word
		# ie word_tag_dict[("dog", "N")] = .4 if there's a .4 chance dog is a noun

		# we put the first and last tags as identical lists of 3, so the matrix will
		# always be of stable dimensions
		prob_matrix = numpy.zeros((len(s)-2,3))
		tag_matrix = [[first_tag]]
		# first, make matrix of all possible tags....
		for i in range(len(s)-2):
			w = s[i+1]
			
			pos_tags = self.babyTagger.quick_tag(w)[2]
			probs = [pos_tags[k] for k in np.arange(0,len(pos_tags),2)]
			tags = [pos_tags[k] for k in np.arange(1,len(pos_tags),2)]
			tag_matrix.append(tags)
			for j in range(3):
				word_tag_dict[(w,tags[j])] = probs[j]

	
		# for the first unknown, we use the MIDDLE trigram probs given the unambiguous first tag,
		# and the sum over all possible following tags
		for i in range(3):
			# just so I know what the variables are....
			word = s[1]
			this_tag = tag_matrix[1][i]
			# EMission... chance that a given word is that tag
			em = word_tag_dict[(w, this_tag)]
			# TRansition... sum of all chances that this is the tag, given possible starts and 3rd states
			tr = sum([self.mommaTagger.prob_middle((first_tag, next_tag), this_tag) for next_tag in tag_matrix[2]]) 
			prob_matrix[0,i] = em * tr

		# for subsequent tags, we use the sum over the probabilites of the previous TWO tags	
		for i in range(1,len(s)-2):
			word = s[i]
			pos_prev = [(laster, last) for laster in tag_matrix[i-1] for last in tag_matrix[i-2]]
			for j in range(3):
				this_tag = tag_matrix[i][j]
				em = word_tag_dict[(w, this_tag)]
				tr = self.mommaTagger.prob_last(pos_prev[j], this_tag)
				prob_matrix[i,j] = tr * em * sum(prob_matrix.sum(i-1)) 

		# okay! now we have a matrix of all possible tag sequences. We just go through the matrix and pick
		# the best from each column and find the corresponding tag
		tag_matrix = tag_matrix[1:]
		gap_guess = []
		for i in range(len(s)-2):
			guess_index = np.argmax(prob_matrix[i])
			gap_guess.append((tag_matrix[i][guess_index], prob_matrix[i,guess_index]))
		print(gap_guess)


		
		


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
	
	