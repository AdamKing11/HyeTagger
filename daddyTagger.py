import re, sys, nltk
import numpy as np
from random import shuffle

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

#from lib import word_features, bclass_cross
from lib.word_features import *
from lib.bclass_cross import *
from lib.test_eval import *
from lib.general import *
from babyTagger import *
from mommaTagger import *

class daddyTagger:
	
	babyTagger = 0
	mommaTagger = 0

	def __init__(self, bt, mt):
		"""
		load in the sub-taggers from the other classes
		1st is based on unambiguous tokens
		2nd is based off of sentences with ONLY unambigous tokens
		"""
		self.babyTagger = bt
		self.mommaTagger = mt

	def say_hello(self):
		"""
		display information about the taggers
		"""
		
		print("Baby Tagger:")
		print("\tAccuracy:", self.babyTagger.accuracy)
		print("\tUnambigous Tokens:", len(self.babyTagger.Unambig))
		print("\tAmbigous Tokens:", len(self.babyTagger.ambig))
		#print("\tTag types:", len(self.babyTagger.all_tags))

		print("Momma Tagger:")
		print("\tTag types:", len(self.mommaTagger.all_tags))
		print("\tTrigram types:", len(self.mommaTagger.mid_prob))
		print("\tTotal trigrams:", self.mommaTagger.total_trigrams)

	def tag(self, s, threshold = 1., fc = True, morph_weight = 1., syn_weight = 1., mc = 4):
		"""
			Tag a sentence! (duh)
			We first go through and find all unambiguous words in the sentence
			then, we use those unambiguous words to give us syntactic context 
			(i.e. tag trigams) information to help break ties for ambiguous tokens
			then we do the same for totally novel words
		"""
		
		# if we've been passed a 'morph_weight' of 0, then we consider ALL tags for all words
		# EXCEPT the "closed" tag types - PUNC, N/A, START, END
		if morph_weight == 0:
			mc = len(self.mommaTagger.all_tags)-4
			
		# get rid of any blank words we may have been given....
		s = [w for w in s if w != ""]
		# build an empty list of length s
		tagged = ["" for _ in s]
	
		# first pass, find and tag all unambiguous tokens or reeeeallly high prob
		# unknown words
		for i in range(len(s)):
			# fold case?
			if fc:
				w = s[i].lower()
			else:
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
				
				gap_guesses = self.gap_fill(gap, morph_weight = morph_weight, \
					syn_weight = syn_weight, mc = mc)
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

	def gap_fill(self, s, morph_weight = 1., syn_weight = 1., mc = 5):
		"""
		a means of finding the best fit of tags for an unknown sequence
		we start with KNOWING the first and last word/tag pair (but we could always do
		it with unknowns...) and then find the highest prob sequence of tags for the given
		words

		we do this via the product of the probability of a tag|word (from the morphological
		Bayesian classifier) and the SUM over all tag trigrams that COULD make up a tag sequence
		
		'mc' for maximum tag candidates we allow per word
		"""

		first_tag = self.babyTagger.quick_tag(s[0])[2][1]
		final_tag = self.babyTagger.quick_tag(s[-1])[2][1]
		
		
		# dictionary for storing the probability a given word has a particular tag
		# because we're going to be copying some tags for a given word, this will be 1/3
		# for ambiguous and unambiguous tokens to indicate that any tag is possible
		word_tag_dict = {}
		
		prob_matrix = numpy.zeros((len(s),mc))
		tag_matrix = []

		for i in range(len(s)):
			w = s[i]
			
			pos_tags = self.babyTagger.quick_tag(w, max_guess=mc)[2]
			probs = [pos_tags[k] for k in np.arange(0,len(pos_tags),2)]
			tags = [pos_tags[k] for k in np.arange(1,len(pos_tags),2)]

			if len(tags) == 1:	
				# if it's an unambigous word, ie the first and last
				# we just copy the SINGLE tag multiple times
				tag_matrix.append([tags[0] for _ in range(mc)])
				word_tag_dict[(w, tags[0])] = 1./mc
			elif len(tags) == 2:
				# if it's an ambiguous token with 2 tags, just copy the last one
				tag_list = [tags[0]]
				word_tag_dict[(w, tags[0])] = 1./mc
				for j in range(1,mc):
					word_tag_dict[(w, tags[1])] = 1./mc
					tag_list.append(tags[1])
				tag_matrix.append(tag_list)
			else:
				tag_list = tags
				for j in range(len(tags)):
					word_tag_dict[(w, tags[j])] = probs[j]
				# if we have a prob distribution BUT we still have fewer possible 
				# tags than we'd like, just copy the LAST one
				if len(tags) < mc:
					for j in range(len(tags),mc):
						tag_list.append(tags[len(tags)-1])
				tag_matrix.append(tag_list)


		for i in range(1,len(s)-1):
			w = s[i]
			# get all possible tags for previous and following
			# and by all possible, I mean, all the tags that the other classifiers have possibly
			# assigned to the previous and following words 
			pos_prev = tag_matrix[i-1]
			pos_next = tag_matrix[i+1]
			pos_contexts = [(p, n) for p in pos_prev for n in pos_next]

			for j in range(mc):
				# we sum over the probabilities for ALL possible contexts (previous and following
				#	tags given the current POSSIBLE tag)
				tag = tag_matrix[i][j]
				# get the sum of the probabilities that we have this tag for all contexts....
				# since we're going to regularize it later, no need to get avg....
				tr = sum([word_tag_dict[(s[i-1]), c[0]] * word_tag_dict[(s[i+1]), c[1]] * \
					self.mommaTagger.mid_prob[(c[0], tag, c[1])] for c in pos_contexts])
				# hella complicated - go get the probability of a tag, we SUM the probabilities of
				# the trigrams TIMES the probability of the the tags of the context words (previous
				# and following) given the word
				# so, probability we get T2 for W2: "w1_t1 W2_T2 w3_t3"  is: p(t1|w1) * p(t3|w3) * 
				# p(t1,T2,t3) for all possible w/t combinations for both sides
				em = word_tag_dict[(w,tag)]
				##### changing from product to SUM
				# with the SUM, we can set either weight to 0 and still use the other....
				# because we normalize it for each word, doesn't matter the total value, just
				# relative for each tag
				prob_matrix[i,j] = (em * morph_weight) + (tr * syn_weight)
				# now we normalize the probs...
			for j in range(mc):
				# make sure we've assigned at least SOME probability to one of the tags...
				if prob_matrix[i,].sum() > 0:
					prob_matrix[i,j] = prob_matrix[i,j] / prob_matrix[i,].sum()
				# if no tag has ANY probability, then 1) we have a 0 count for the 
				# trigrams AND the morphological probability is 0.... pick a random 
				# tag and give it some prob
				else:
					rand_tag = np.random.randint(mc)
					# we want to show that we just guessed randomly, low confidence
					prob_matrix[i,rand_tag] = 1/mc
					
				
		gap_guess = []
		for i in range(1,len(s)-1):
			guess_index = np.argmax(prob_matrix[i,])
			confidence = prob_matrix[i,guess_index]
			gap_guess.append((tag_matrix[i][guess_index], confidence))
		
		return gap_guess

	def tag_corpus(self, corpus, verbose = True):
		"""
		not too clean yet...
		go through one of our corpora - EANC or wikipedia - and tag it
		we then save the results in 3 files - one with just the sentences, one 
		with just a sequence of tags and one with each word tagged (w_t)
		"""
		running = []
		goldsX = []
		goldsY = []
		golds = 0


	# for saving our silver boys...
		with open("golds/" + corpus + ".silver.X.txt", "w") as wF:
			pass
		with open("golds/" + corpus + ".silver.Y.txt", "w") as wF:
			pass
		with open("golds/" + corpus + ".silver.M.txt", "w") as wF:
			pass
	
		if verbose:
			print("\n" + "-" * 10 + corpus + "-" * 10 + "\n")
	# let's load in some sentences....
		with open(corpus + ".READY.txt", "r") as rF:
			i = 0
			for line in rF:
				line = line.rstrip()
				s = line.rsplit(" ")
			# get rid of any " " values that may be in there, messing up the stuff
			# damn gremlins
			# mogwai
				s = [w for w in s if w != ""]

				if len(s) > 7:
					i += 1
					if verbose:
						print(golds, "of", i, "sentences.", end="\r")

					if i < 0:
						continue
					if i > 10000:
						pass	# do them all!!!!
						break
					t, g = daddy.tag(s)
					conf = 1.
					if len(g) > 1:
					# if there are some guesses, let's check to see the mean 
					# confidence in those guesses
						conf = sum(s[1] for s in g)/len(g)
					else:
						conf = 1.0
					if not np.isnan(conf):
						running.append(conf)
					tagged_s = ""

					if conf >= 1.0:
						for j in range(len(s)):
							tagged_s += s[j] + "_" + t[j] + " "
						tagged_s = tagged_s[0:-1]

						g_X = s[0]
						g_Y = t[0]
						golds+=1
						for j in range(1, len(s)):
							g_X += "\t" + s[j]
							g_Y += "\t" + t[j]
						with open("golds/" + corpus + ".silver.X.txt", "a") as wF:
							wF.write(g_X + "\n")
						with open("golds/" + corpus + ".silver.Y.txt", "a") as wF:
							wF.write(g_Y + "\n")
						with open("golds/" + corpus + ".silver.M.txt", "a") as wF:
							wF.write(tagged_s + "\n")

		if verbose:		
			print("Over", len(running), "sentences, we have a mean confidence of", np.mean(running), \
				"with a variance of", np.var(running))	
			print(golds, "silver sentences in the bunch.\n")


if __name__ == "__main__":
	print("Loading the 2 taggers...")
	# load the 2 sub-taggers
	baby = c_load("taggers/b_tagger.t")
	momma = c_load("taggers/m_tagger.t")
	# make the big one
	#print("Building the final tagger...")
	daddy = daddyTagger(baby, momma)
	#c_save(daddy, "taggers/d_tagger.t")

	daddy.say_hello()
	
	# tag the corpora
	corpus = "hyWiki"
	#corpus = "EANC"
	#daddy.tag_corpus(corpus)

		# split our gold data into test and train
	#train, test = split_corpus("hyWiki.golds.txt", ratio = .75, shuf = False)
	train, test = split_corpus("EANC.200.hand.txt", ratio = .5, shuf = False)
	

		# find all words that are in JUST the test data
	_, train_u, test_u = find_unique_words(test, train, verbose = False)
		# find a count of trigrams in the test data
	tg = count_trigrams(test, verbose = False)

	#baby.limit_and_rebuild(500)
	#baby.test_baby_classifier(6)
	baby.forget(unambig_to_f = test_u, verbose = False)
	momma.forget(trigrams_to_f = tg, verbose = False)

	
	print("\n\nDone forgetting, re-building classifier...")
	new_daddy = daddyTagger(baby, momma)
	#c_save(new_daddy, "taggers/nd_tagger.t")
		

	#new_daddy = c_load("taggers/nd_tagger.t")
	new_daddy.say_hello()

	# now, let's try and tag the sentences in 'test'
	score_tagger(test, new_daddy, morph_weight = 1, syn_weight = 1, verbose=False)