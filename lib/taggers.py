import re, sys, nltk
import random
from random import shuffle
import numpy as np

"""
Armenian POS Tagger
Adam King 2016 University of Arizona

A POS tagger that uses Naive Bayesian classifiers (and LSTM language models) to tagger
Armenian language text
	"""

from lib.general import *
from lib.word_features import *
from lib.test_eval import *


###################################################################################################
###################################################################################################
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
    We load in the EANC token file and build a naive Bayesian classifier for UNAMBIGUOUS tokens
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

	def remember(self, unambig_to_r = [], ambig_to_r = [], verbose = True):
		"""
		not really "remembering" per se, just adding some tokens to either list and re-building
		"""
		if verbose:
			print("Going to add", len(unambig_to_r), "unambiguous tokens.")
			print("\t", len(ambig_to_r), "ambiguous tokens.")
		for r in unambig_to_r:
			self.Unambig[r] = [unambig_to_r[r]]

		for r in ambig_to_r:
			self.ambig[r] = ambig_to_r[r]

		if verbose:
			print("Done. New counts:")
			print("\tUnambiguous:", len(self.Unambig))
			print("\tAmbiguous:", len(self.ambig))

		self.build_baby_classifier(verbose = verbose)

	def semisupervised_training(self, word_list, iterations = 3, threshold = .99, verbose = True, \
		folds = 6):
		"""
	goes through the 'word_list' and tests all words with the 'baby' babyTagger
	for all words that beat the 'threshold', we add it to the list of **unambiguous** 
	tokens in the babyTagger and retrain
		"""
		if verbose:
			print("Going to look through", len(word_list), "to find words that", end = " ")
			print("score higher than", threshold, "and add them to the classifier.")
	
	# if we haven't already tested the classfier, do it now
	# so we know the overall accuracy at the beginning
		if self.accuracy == 1:
			self.test_baby_classifier(folds, verbose = verbose)

	# we're going to get the accuracy of the original classifier
	# so we know how to 
		original_accuracy = self.accuracy
	# loop through the iterations
		for i in range(iterations):
			high_scores = {}
			if verbose:
				print("\nIteration", i+1)
				print("\t", len(word_list), "words to check.")
				j = 0
			for w in word_list:
				if verbose:
					j+=1
					print("Checking word ::", j, end="\r")
			# tag the word using the tagger 
				tag_guess = self.quick_tag(w)
			# if it's not in the lexicon already AND it's score beats our threshold
			# add it to the list
	
				if tag_guess[0] == 3 and tag_guess[2][0] >= threshold:
					word = tag_guess[1]
					tag = tag_guess[2][1]
					high_scores[word] = word + "_" + tag
				
		# if we found no words to add, just exit
			if len(high_scores) == 0:
				if verbose:
					print()
					print("No new words found to add. Ending training.")
				break
		
		# now that we have the high-scoring words, we delete them from the orginal list
		# (to avoide adding them again and again) AND we use a random number to determine
		# if we really want to add them, based on the accuracy of the tagger
		# that is, if accuracy of the tagger is 90%, we add 90% of the words
		# if it's 50%, we only add 50%, etc
			if verbose:
				print()
				print("Found", len(high_scores), "words to add.")
				j = 0

		# go through all the high scoring words and generate a randome number 0-1
		# if the number is LESS than the accuracy, keep
		# this way, we don't add too many words to an already bad classifer
			high_score_to_delete = {}
			for w in high_scores:
				del word_list[w]
				keep = np.random.random()
				if keep >= original_accuracy:
					j+=1
					high_score_to_delete[w] = 1

			for w in high_score_to_delete:
				del high_scores[w]
			if verbose:
				print("Randomly removed", j, "words leaving us with", len(high_scores))

		# add those high scoring words to the unambiguous list and retrain
		self.remember(unambig_to_r = high_scores)
	

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
		elif re.search("[0-9]", word):
			# if there's a numeral in it, just assume that it's a numeral
			tagged_entry = (3, word, (1, "NUM"))	
		else:
			# we look at the top 'max_guess' best guesses
			guess_matrix = [(0,0) for _ in range(max_guess)]
			# get a list of tags and probabilities
			# because we only have the TOKEN, assume the token = lemma
			guess = self.babyTagger.prob_classify(addFeatures(word,word))
			for g in sorted(guess.samples(), key=guess.prob, \
                    reverse=True)[:max_guess]:
			## we get the guesses and MULTIPLY by the accuracy of our classifer
			# EDIT - we don't take the accuracy in to account here, we do that outside
				guess_prob = guess.prob(g)# * self.accuracy
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
			self.tag_trigrams, self.all_tags = self.read_hand_tagged(tagged_c, verbose)
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
				for k in range(2, len(l)):
					j += 1
					# get the tag 2 ago, 1 ago and current
					# probably a better way to do this, rather than re-calc every time....
					last_tag = split_tagged_lemma(l[k-2])[1]
					all_tags.add(last_tag)
					
					tag = split_tagged_lemma(l[k-1])[1]
					all_tags.add(tag)
					
					next_tag = split_tagged_lemma(l[k])[1]
					all_tags.add(next_tag)

					tg = (last_tag, tag, next_tag)
					
					if tg in trigrams:
						trigrams[tg] += 1
					else:
						trigrams[tg] = 1
		if verbose:
			print("Read in", i, "sentences and found", j, "individual trigrams with", len(all_tags), \
				"total tag types.")
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


#######################################################################################
#######################################################################################
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
