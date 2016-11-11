import re, sys, nltk
from random import shuffle
from bclass_cross import *
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


class babyTagger:

	Unambig = {}	# dictionary for UNambiguous tokens
	ambig = {}		# dictionary for ambiguous tokens

	tag_count = {}	# for storing the types of tags we have and their counts
	babyTagger = 0

	def __init__(self, token_file, verbose = True):
		"""
		create the tagger
		"""
		_, self.Unambig, self.ambig, _ = self.read_EANC(token_file, verbose = verbose)
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
    	# we take our unambiguous tokens and get them formatted for training
    	# we're also going to save the tag types and counts, just for posterity
		training_set, self.tag_count = format_training_data(self.Unambig)     
    			
		# shuffle the order of the list
		shuffle(training_set)

		if verbose:
			print("\nNow training a Naive Bayesisan Classifier on", \
				len(training_set), "unambiguous tokens.")
		self.babyTagger = nltk.NaiveBayesClassifier.train(training_set)

	def test_baby_classifier(self, k, verbose = True):
		"""
		does a quick k-fold test on the classifier we've built... or at least on
		the features/data we've used to make the classifier....
		"""
		training_set, _ = format_training_data(self.Unambig)
				
		# shuffle the order of the list
		shuffle(training_set)
		cross_validation(training_set, k, verbose = verbose)


	def quick_tag(self, word):
		"""
		quick and dirty tagger
			if word is in our unambiguous list - tag as that unambiguous tag
			if word in ambiguous list - tag EQUIPROBABLE as any of its tags
			if in neither list, use the Bayesian classifier to GUESS and score
		"""
		
		tagged_entry = ()
		# UNAMBIGOUS WORDS
		if word in self.Unambig:
			tagged_entry = (1, word, (1, split_tagged_lemma(self.Unambig[word][0])[1]))
			
		# AMBIGUOUS WORDS
		elif word in self.ambig:
			# get all the tags from the various lemmas
			tags = set([split_tagged_lemma(lemma)[1] for lemma in self.ambig[word]])
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
			for g in sorted(guess.samples(), key=guess.prob, \
                    reverse=True)[:3]:
				guess_matrix.append((guess.prob(g), g))
			# we'll save the best 3 results   
			guess_matrix = sorted(guess_matrix, reverse=True)[:3]
            
            # flatten the list of lists
			guess_matrix = tuple([x for y in guess_matrix for x in y])
			tagged_entry = (3, word, guess_matrix)
		return tagged_entry
	
	def quick_tag_sentence(self, s):
		"""
	takes a sentence, i.e. a list of words, and returns a list of tags and
	our confidence in said tags
		"""	
		tagged_s = []
		# for holding the best probability guess for the tokens
		scores = []
		for w in s:
			t = self.quick_tag(w)
			tagged_s.append(t)
			scores.append(t[2][0])
		return tagged_s, numpy.mean(scores)


if __name__ == "__main__":
   	babyTag = babyTagger("EANC_tokens.txt")
   	#babyTag = babyTagger("50000.EANC.txt")
   	#babyTag.test_baby_classifier(4)
   	print(babyTag.quick_tag("բան"))
   	print(babyTag.quick_tag("համար"))
   	print(babyTag.quick_tag("ջահանը"))

   	#s = "<s> Դուք պետք է հավաստեք , որ ձեր ներլցած ֆայլը ոչ մի հեղինակային իրավունք չի խախտում ։ </s>"
   	
   	with open("hyWiki_sub.txt") as rF:
   		i = 0
   		for s in rF:
   			s = s.rstrip()
   			s = s.rsplit(" ")
   			ts, score = babyTag.quick_tag_sentence(s)
   			j = 0
   			for w in ts:
   				j+=1
   				#print(i,"\t",w)
   			print(i, "- Avg. Score ::", score)
   			i+=1
   			if i > 20:
   				break
   	
   	

   	