import re, sys, nltk
from random import shuffle
"""
Armenian POS Tagger
Adam King 2016 University of Arizona

A POS tagger that uses Naive Bayesian classifiers (and LSTM language models) to tagger
Armenian language text

class hyTagger:
	morphBayes - classifier trained on unambiguous tokens from EANC
	synBayes - classifier built on tag trigrams

"""

from word_features import *


class hyTagger:

	unambiguous = []
	ambiguous = []

	babyTagger = 0


	def __init__(self, token_file, verbose = True):
		"""
		create the tagger
		"""
		_, self.unambigous, self.ambiguous, _ = self.read_EANC(token_file, verbose = verbose)
		self.build_baby_classifier(verbose = verbose)

		

	def read_EANC(self, token_file, verbose = True):
		"""
    Reads in a list of tokens we've culled from EANC and returns a list of:
    TOKEN[token] - (unigram_count, TAGS, DEFS)
    UNIQUE[token] - [TAG]
    AMBIGUOUS[token] - [TAGS]
    NO_POS[token] - []  
    	"""
		tokens = {}
		unique = {}
		ambiguous = {}
		no_pos = {}
		if verbose:
			print("Reading in tokens from", token_file)
		with open(token_file, "r") as rF:
			i = 0
			for line in rF:
				i+=1
				if verbose:
					print("\tReading token - ", i, end="\r")
				line = line.rstrip()
				l = line.rsplit("\t")
				# if it's a blank line for whatever reason, SKIP
				if l[0] == "":
					continue
            	# the token in question is first thing on line... duh
				token = l[0]
				try:
					tokens[token] = (l[1], l[2], l[3])
				except:
					# if we can't split it, it's because there's nothing, ie no POS
					tokens[token] = (l[1], [], [])
            	# if there is no token, just skip skip skip
				tags = re.sub("[\['\]]", "", l[2]).rsplit(",") 
				# here, we get all the tags for the various lemmas associated with the
				# token         
				tag_set = set([])
				for t in tags:
					if t != "":
						tag_set.add(t)
				# now that we've gotten all the tags from the various lemmas,
				# decide which group to put the token in
				if len(tag_set) == 0:
					no_pos[token] = []
				elif len(tag_set) > 1:
					ambiguous[token] = list(tag_set)
				else:
					unique[token] = list(tag_set)
        # return all lists
		if verbose:
			print("\nTotal tokens:", len(tokens))
			print("\tUnambiguous:",len(unique))
			print("\tAmbiguous:",len(ambiguous))
			print("\tNo Label:",len(no_pos))

		return tokens, unique, ambiguous, no_pos

	def build_baby_classifier(self, verbose = True):
		"""
    We load in the EANC token file and build a classifier for UNAMBIGUOUS tokens
    	"""
    	# create the training 
		training_set = []     
    	# put the unambiguous tokens into the right format for the classifier
		for token in self.unambigous:
			##### for continuity, the unambigous dictionary stores LISTS
			# make sure we do the 0th element of the length 1 list
			lemma, tag = split_tagged_lemma(self.unambigous[token][0]) 
		
        # for some reason, some BAD tags exist in EANC and they all are in
        # all lower so the upper() part sorts that out....
			features = addFeatures(token, lemma)
			if tag == tag.upper():
				training_set.append((features,tag))
				#all_tags.add(tag)

		# shuffle the order of the list
		shuffle(training_set)

		if verbose:
			print("\nNow training a Naive Bayesisan Classifier on", \
				len(training_set), "unambiguous tokens.")
		self.babyTagger =  nltk.NaiveBayesClassifier.train(training_set)

	def quick_tag(self, word):
		"""
		quick and dirty tagger
			if word is in our unambiguous list - tag as that unambiguous tag
			if word in ambiguous list - tag EQUIPROBABLE as any of its tags
			if in neither list, use the Bayesian classifier to GUESS and score
		"""
		# UNAMBIGOUS WORDS
		print(word in self.unambiguous)
		for w in self.unambigous:
			if w==word:
				print("goodbye")
				print(w)

		if word in self.unambiguous:
			print("hello")
			tagged_entry = (1, w, (1, split_tagged_lemma(unambig[w][0])[1]))
			
		# AMBIGUOUS WORDS
		elif word in self.ambiguous:
			# get all the tags from the various lemmas
			tags = set([split_tagged_lemma(lemma)[1] for lemma in self.ambiguous[word]])
            # make it a memory efficient, sorted tuple of tags
            # doing this so all words of the same ambiguity can be grouped later (maybe)
			tags = tuple(sorted(tags))
            
			# for ambiguous tokens, we're going to assume that all tags are equiprobable
			guess = []
			guess_prob = (1./len(tags))
			# we go through all the possible tags and put it in the list, with the probability
			for t in tags:
				guess.append(guess_prob)
				guess.append(t)
			# make it a tuple #memory #bigO
			guess = tuple(guess)
            
			if guess_prob < 1:
				tagged_entry = (2, word, guess)
			else:
				# b/c sometimes tokens with amibiguous lemmas have lemmas of the
            	# same POS, we can just call them unambiguous...
				tagged_entry = (1, word, (1, guess[1]))
		
		# UNKNOWN WORDS
		else:
			# we look at the top 3 best guesses
			guess_matrix = [(0,0) for _ in range(3)]
			# get a list of tags and probabilities
			# because we only have the TOKEN, assume the token = lemma
			guess = self.babyTagger.prob_classify(addFeatures(word,word))
			i = 0
			for g in sorted(guess.samples(), key=guess.prob, \
                    reverse=True)[:3]:
				#guess_matrix[i] = (guess.prob(g), g)
				guess_matrix.append((guess.prob(g), g))
				i += 1
            # we'll save the best 3 results   
			guess_matrix = sorted(guess_matrix, reverse=True)[:3]
                # flatten the list of lists
			guess_matrix = tuple([x for y in guess_matrix for x in y])
			tagged_entry = (3, word, guess_matrix)
		return tagged_entry
			


if __name__ == "__main__":
   	#babyTag = hyTagger("EANC_tokens.txt")
   	babyTag = hyTagger("50000.EANC.txt")
   	print(babyTag.quick_tag("բան"))
   	print(babyTag.quick_tag("համար"))
   	print(babyTag.quick_tag("ջահանը"))