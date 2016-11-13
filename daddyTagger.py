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


		# returns a measure of how confident we are with guessed words
		# list of tuples, (word_location, confidence)
		guess_info = []

		# second pass, fill in any gaps
		# how we do this is we find the last good tag before a gap and get a substring
		# until the next good tag
		# then we pass that substring to a function that finds the optimal set of tags
		start_gap = -1
		end_gap = -1
		for i in range(1,len(s)):
			# if we're at a blank tag and the last one wasn't blank, we're at the 
			# start of a gap
			if tagged[i] == "" and tagged[i-1] != "":
				start_gap = i-1
				end_gap = -1
			# if we already know we're in a gap and now we've seen an unambiguous word,
			# we grab the gap, tag it and then reset
			if start_gap >= 0 and tagged[i] != "":
				end_gap = i+1
				gap = s[start_gap:end_gap]
				# we'll now send the gap to our happy, little gap guess function thing
				# it will return the best guess and a measure of probability that that
				# guess is right
				gap_guesses = self.gap_fill(gap)
				j = start_gap + 1
				for g in gap_guesses:
					# fill in our sequence of tags with the winner for this word
					tagged[j] = g[0]
					# save the position of the guess along with the score
					guess_info.append((j,g[1]))
					j += 1 
				start_gap = -1
				end_gap = -1

		return tagged, guess_info

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
		final_tag = self.babyTagger.quick_tag(s[-1])[2][1]
		
		word_tag_dict = {}	# probability of a TAG given a word
		# ie word_tag_dict[("dog", "N")] = .4 if there's a .4 chance dog is a noun

		# we put the first and last tags as identical lists of 3, so the matrix will
		# always be of stable dimensions
		prob_matrix = numpy.zeros((len(s)-2,3))
		tag_matrix = [[first_tag for _ in range(3)]]
		# first, make matrix of all possible tags....
		for i in range(len(s)-2):
			w = s[i+1]
			
			pos_tags = self.babyTagger.quick_tag(w)[2]
			probs = [pos_tags[k] for k in np.arange(0,len(pos_tags),2)]
			tags = [pos_tags[k] for k in np.arange(1,len(pos_tags),2)]
			# if we have fewer than 3 possible tags for a given word, just copy
			# the last one. because we're summing over all possibilities (#Forward-Backward)
			# the extra tag will just fall out in the wash... I think :/
			if len(tags) < 3:
				tags.append(tags[1])
				probs.append(probs[1])
			tag_matrix.append(tags)
			for j in range(3):
				word_tag_dict[(w,tags[j])] = probs[j]
			
		# for the first unknown, we use the MIDDLE trigram probs given the unambiguous first tag,
		# and the sum over all possible following tags
		for i in range(3):
			# just so I know what the variables are....
			w = s[1]
			this_tag = tag_matrix[1][i]
			# EMission... chance that a given word is that tag
			em = word_tag_dict[(w, this_tag)]
			# TRansition... sum of all chances that this is the tag, given possible starts and 3rd states
			tr = sum([self.mommaTagger.prob_middle((first_tag, next_tag), \
				this_tag) for next_tag in tag_matrix[1]])
			prob_matrix[0,i] = em * tr

		# for subsequent tags, we use the sum over the probabilites of the previous TWO tags	
		for i in range(2,len(s)-2):
			w = s[i]
			# build list of all possible previous 2 tags
			pos_prev = [(laster, last) for laster in tag_matrix[i-1] for last in tag_matrix[i-2]]
			for j in range(3):
				this_tag = tag_matrix[i][j]
				em = word_tag_dict[(w, this_tag)]
				#tr = self.mommaTagger.prob_last(pos_prev[j], this_tag)
				tr = sum([self.mommaTagger.prob_last(pp, this_tag) for pp in pos_prev])

				# i-1 because the tag_matrix is 1 bigger than the prob_matrix.....
				prob_matrix[i-1,j] = tr * em# * prob_matrix[i-2,].sum()
				
		# for the last tag of the gap, we will go back to predicting the middle 
		# tag based off of the previous guesses and the final tag (which is certain)
		if len(s) >= 3:
			for i in range(3):
				w = s[-2]
				this_tag = tag_matrix[-1][i]
				em = word_tag_dict[(w, this_tag)]
				tr = sum([self.mommaTagger.prob_middle((last_tag, final_tag), \
					this_tag) for last_tag in tag_matrix[-2]])
				prob_matrix[-1,i] = tr * em * prob_matrix[-1,].sum()
				
		# okay! now we have a matrix of all possible tag sequences. We just go through the matrix and pick
		# the best from each column and find the corresponding tag
		tag_matrix = tag_matrix[1:]
		gap_guess = []
		for i in range(len(s)-2):
			guess_index = np.argmax(prob_matrix[i])
			gap_guess.append((tag_matrix[i][guess_index], prob_matrix[i,guess_index]))
		return gap_guess


	

if __name__ == "__main__":
	# load the 2 sub-taggers
	b = c_load("b_tagger.t")
	m = c_load("m_tagger.t")
	# make the big one
	daddy = daddyTagger(b, m)

	# let's load in some sentences....
	with open("hyWiki_sub.txt", "r") as rF:
		i = 0
		for line in rF:
			line = line.rstrip()
			s = line.rsplit(" ")
			if len(s) > 7:
				i += 1
				if i > 3:
					break
				print(i, line,"\n")
				t, g = daddy.tag(s)
				print(t,"\n", g, "\n\n")

	#s = "<s> Դուք պետք է հավաստեք , որ ձեր ներլցած ֆայլը ոչ մի հեղինակային իրավունք չի խախտում ։ </s>"
	s = "<s> այս էջի կոդը ։ </s>"
	s = s.rsplit(" ")
	#print(s, "\n")
	#t,_ = daddy.tag(s)
	#print(t)
	#print(daddy.mommaTagger.prob_last(("N", "N"), "N"))
	#print(daddy.mommaTagger.prob_last(("N", "V"), "N"))
	#print(daddy.mommaTagger.prob_middle(("N", "N"), "N"))
	#print(daddy.mommaTagger.prob_middle(("N", "N"), "V"))
	#print(daddy.mommaTagger.tag_trigrams[("N","N", "N")])
	#print(daddy.mommaTagger.tag_trigrams[("N","V", "N")])