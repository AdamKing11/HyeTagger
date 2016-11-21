import re, sys, nltk
import random
"""
Armenian POS Tagger
Adam King 2016 University of Arizona

A POS tagger that uses Naive Bayesian classifiers (and LSTM language models) to tagger
Armenian language text
	"""

from lib.general import *
from lib.word_features import *
from lib.bclass_cross import *
from lib.test_eval import *


class babyTagger:

	Unambig = {}	# dictionary for UNambiguous tokens
	ambig = {}		# dictionary for ambiguous tokens

	unknown = {}

	tag_count = {}	# for storing the types of tags we have and their counts
					# note, this only holds counts for "open" classes, tags
					# that we train the classifier on, ie not tags we EXPECT
					# to find new words for, ie punctuation, sentence start/end, etc

	babyTagger = 0	# just a place holder

	accuracy = 1. 	# for storing the accuracy of our tagger,
					# in case we want to indicate our trust in the scores given
					# by the classifier. ASSUME it's 1, we change it iff we
					# test it

	all_tags = {}	# to see which tags we're actually working on predicting
					# also, get a count of each tag... maybe for a prior prob

	def __init__(self, tokens, read_from_file = True, verbose = True):
		"""
		create the tagger
		we can either have it read in the EANC tokens OR pass it a dictionary
		of tokens, we assume to be unambiguous
		"""
		if read_from_file:
			self.Unambig, self.ambig, _, _, self.all_tags = read_EANC(tokens, verbose = verbose)
		else:
			# because unambiguous tokens are always in a list of size 1
			for t in tokens:
				self.Unambig[t] = [tokens[t]]
		# read_EANC function in the libs
		self.build_baby_classifier(verbose = verbose)
	

	def build_baby_classifier(self, verbose = True):
		"""
    We load in the EANC token file and build a classifier for UNAMBIGUOUS tokens
    	"""
    	# we take our unambiguous tokens and get them formatted for training
    	# we're also going to save the tag types and counts, just for posterity
		training_set, self.tag_count = format_training_data(self.Unambig)     
    			
		# shuffle the order of the list
		random.shuffle(training_set)

		if verbose:
			print("\nNow training a Naive Bayesisan Classifier on", \
				len(training_set), "unambiguous tokens.")
		self.babyTagger = nltk.NaiveBayesClassifier.train(training_set)

	def limit_and_rebuild(self, tag_max, tag_types = None, verbose = True):
		"""
		take a classifier, shrink down the number of ambiguous tokens to a 
		particular max count per tag and rebuild
		"""

		if verbose:
			print("Shrinking the dictionary of unambiguous tokens.")
			print("Starting with", len(self.Unambig), "unambiguous tokens.")
		
		self.Unambig = shrink_token_dict(self.Unambig, tag_max = tag_max, \
			tag_types = tag_types)

		if verbose:
			print("\tDone. Now, we have", len(self.Unambig), "tokens. Re-training...")

		self.build_baby_classifier(verbose = verbose)	



	def test_baby_classifier(self, k, verbose = True):
		"""
		does a quick k-fold test on the classifier we've built... or at least on
		the features/data we've used to make the classifier....
		"""
		training_set, _ = format_training_data(self.Unambig)
				
		# shuffle the order of the list
		random.shuffle(training_set)
		self.accuracy, _ = cross_validation(training_set, k, verbose = verbose)


	def forget(self, unambig_to_f = None, ambig_to_f = None, verbose = True):
		"""
		million dollar idea, dollars and donuts
		we're going to "forget" certain unambiguous and ambiguous tokens we've seen 
		and retrain
		this way, we can test on "gold" sentences we've found by looking up every word
		and getting a sentence of ONLY unambiguos tokens
		"""

		if unambig_to_f == None:
			unambig_to_f = []
		if ambig_to_f == None:
			ambig_to_f = []

		if verbose:
			print("Now **forgetting**", len(unambig_to_f), "unambiguous tokens and",\
				len(ambig_to_f), "ambiguous tokens.")
			print("Previous unambiguos token count:", len(self.Unambig))
		for f in unambig_to_f:
			if f in self.Unambig:
				del self.Unambig[f]
		if verbose:
			print("\tNew unambiguos count:", len(self.Unambig))
			print("Previous ambiguos token count:", len(self.ambig))
		for f in ambig_to_f:
			if f in self.ambig:
				del self.ambig[f]
		if verbose:
			print("\tNew ambiguous token count:", len(self.ambig))
			print("Retraining....")

		self.build_baby_classifier(verbose = verbose)

	def quick_tag(self, word, fc = True, max_guess = 3):
		"""
		quick and dirty tagger
			if word is in our unambiguous list - tag as that unambiguous tag
			if word in ambiguous list - tag EQUIPROBABLE as any of its tags
			if in neither list, use the Bayesian classifier to GUESS and score
		"""
		
		if len(word) == 0:
			return "N/A"

		if fc:
			word = word.lower()

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
			# we look at the top 'max_guess' best guesses
			guess_matrix = [(0,0) for _ in range(max_guess)]
			# get a list of tags and probabilities
			# because we only have the TOKEN, assume the token = lemma
			guess = self.babyTagger.prob_classify(addFeatures(word,word))
			for g in sorted(guess.samples(), key=guess.prob, \
                    reverse=True)[:max_guess]:
			## we get the guesses and MULTIPLY by the accuracy of our classifer
				guess_prob = guess.prob(g) * self.accuracy
				guess_matrix.append((guess_prob, g))
			# we'll save the best 3 results   
			guess_matrix = sorted(guess_matrix, reverse=True)[:max_guess]
            
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
			if w == "":
				continue
			t = self.quick_tag(w)
			tagged_s.append(t)
			scores.append(t[2][0])
		return tagged_s, numpy.mean(scores)

	def quick_tag_corpus(self, hyCorpora, outfile, total_s = 100, \
			min_w = 8, min_score = 0, verbose = True):
		"""
		go through a CLEANED corpus file and tag the sentences using the tagger
		"""

		rF = open(hyCorpora, "r")
		wF = open(outfile, "w")

		if verbose:
			print("\nNow tagging", hyCorpora)
			if total_s == 0:
				print("Reading ALL sentences of length", min_w)
			else:
				print("Reading", total_s, "sentences of length", min_w)

		i = 0
		for line in rF:
			line = line.rstrip()
			# our function takes a sentence as a list of words...
			s = line.rsplit(" ")
			# make sure the sentence is long enough...
			if len(s) < min_w:
				continue
			# make sure we don't do too many sentences....
			# if 0, then do all
			if i >= total_s and total_s != 0:
				break
			if verbose:
				print("\tReading and tagging sentence #" + str(i), end="\r")
			# tag sentence
			tagged_sentence, mean_score = self.quick_tag_sentence(s)

			if mean_score >= min_score:
				for w in tagged_sentence:
					writeString = w[1]
				# loop through the tags and their scores
					for t in w[2]:
						writeString += "\t" + str(t)
					wF.write(writeString + "\n")
					
				wF.write(str(mean_score) + "\t" + str(len(s)) + "\n\n")
				i += 1

		if verbose:
			print()
			print("\nDone reading file. Tagged and saved", i, "sentences.\nClosing now.")
		rF.close()
		wF.close()

if __name__ == "__main__":
	#babyTag = babyTagger("10000.txt")
	#babyTag = babyTagger("EANC_tokens.txt")
	#babyTag.test_baby_classifier(6)

	#baby = c_load("taggers/b_tagger.t")
	#train, test = split_corpus("EANC.golds.txt", ratio = .5)
	#b, train_u, test_u = find_unique_words(test, train)

	#for ts in test:
	#	s = [split_tagged_lemma(w)[0] for w in ts]
	#	tagged, score = babyTag.quick_tag_sentence(s)
	#	print(score)


	baby = babyTagger("EANC_tokens.txt")
	#baby.test_baby_classifier(6)

	#c_save(baby, "taggers/b_tagger.t")
	#baby = c_load("taggers/b_tagger.t")
	
	#baby.quick_tag_corpus("EANC.READY.txt", "hand_gold/g.csv", min_score = .95,
	#	min_w = 10, total_s = 300)
	#baby.shrink_token_dict(100)
	#baby.test_baby_classifier
	
	
	"""
	baby.forget(unambig_to_f = test_u)


	for ts in test:
		s = [split_tagged_lemma(w)[0] for w in ts]
		tagged, score = babyTag.quick_tag_sentence(s)
		for g in tagged:
			if g[0] != 1:
				print("\nWe don't know:", g[1], g[2])
   	#babyTag = babyTagger("EANC_tokens.txt")
   	#babyTag = c_load("taggers/b_tagger.t")
   	#babyTag.test_baby_classifier(4)
   	#print(babyTag.all_tags)
   	
   	#c_save(babyTag, "taggers/b_tagger.t")
   	
   	#print(babyTag.quick_tag("բան"))
   	#print(babyTag.quick_tag("համար"))
   	#print(babyTag.quick_tag("ջահանը"))

   	#s = "<s> Դուք պետք է հավաստեք , որ ձեր ներլցած ֆայլը ոչ մի հեղինակային իրավունք չի խախտում ։ </s>"
   	#s = s.rsplit(" ")
   	#print(babyTag.quick_tag_sentence(s))
   	#babyTag.quick_tag_corpus("hyWiki.READY.txt", "tagged.wiki.txt", total_s=0)
   	#for t in babyTag.all_tags:
   	#	print(t, babyTag.all_tags[t])

   	"""