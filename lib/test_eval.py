import re, sys, random
import numpy as np

from lib.general import *

def split_corpus(corpus_file, ratio = .75, shuf = True):
	"""
	split a list of sentences into two parts with the first
	part taking up the 'ratio' of the original bulk
	"""
	whole_corpus = []
	train = []
	test = []

	with open(corpus_file, "r") as rF:
		for line in rF:
			line = line.rstrip()
			line = line.rsplit("\t")
			whole_corpus.append(line)

	if shuf:
		random.shuffle(whole_corpus)

	train_len = int(len(whole_corpus) * ratio)
	train = whole_corpus[:train_len]
	test = whole_corpus[train_len:]
	return train, test

def read_untagged_corpus(c, fc=True, verbose = True):
	"""
	reads in a corpus where words are separated by " "
	just like the other function, but no tags
	"""
	unigrams = {}
	i = 0
	with open(c, "r") as rf:
		for line in rf:
			i+=1
			if verbose:
				print("Reading in line", i, end="\r")
			line = line.rstrip().rsplit(" ")
			for w in line:
				if fc:
					w = w.lower()
				if w in unigrams:
					unigrams[w] += 1 
				else:
					unigrams[w] = 1
	if verbose:
		print()
	return unigrams

def read_corpus(corpus_file, shuf = True):
	"""
	reads in a corpus file where each sentence is on its own line and
	the format is "w_t w_t w_t..."
	"""
	whole_corpus = []

	with open(corpus_file, "r") as rF:
		for line in rF:
			line = line.rstrip()
			line = line.rsplit("\t")
			whole_corpus.append(line)

	if shuf:
		random.shuffle(whole_corpus)

	return whole_corpus

def get_words(c, fc = True):
	"""
	takes a list of sentences (like from the function above)
	and shoots out a list of words in that corpus
	"""
	lex = {}
	for s in c:
		for w in s:
			word = split_tagged_lemma(w)[0]
			if fc:
				word = word.lower() 
			lex[word] = w
	return lex


def find_unique_words(test, train, verbose = True):
	"""
	return list of words in 'test' and not 'train'
	and a list of words in 'train' and not 'test'
	"""
	test_u = set([])
	train_u = set([])

	both_all = set([])
	test_all = set([])
	train_all = set([])

	# get list of all words in test
	for s in test:
		for w in s:
			word = split_tagged_lemma(w)[0]
			word = word.lower()
			test_all.add(word)
			both_all.add(word)

	# get list of all words in test
	for s in train:
		for w in s:
			word = split_tagged_lemma(w)[0]
			word = word.lower()
			train_all.add(word)
			both_all.add(word)

	for s in test:
		for w in s:
			word, _ = split_tagged_lemma(w)
			word = word.lower()
			if word not in train_all:
				test_u.add(word)

	for s in train:
		for w in s:
			word, _ = split_tagged_lemma(w)
			word = word.lower()
			if word not in test_all:
				train_u.add(word)

	if verbose:
		print("Total word types in both ::", len(both_all))
		print("Word types in 'test' ::", len(test_all))
		print("Word types in 'train' ::", len(train_all))
		print("\tRatio :: %.2f" % (len(test_all)/len(both_all)))
		print("Unique words in 'test' ::", len(test_u))
		print("Unique words in 'train' ::", len(train_u))
		print("\tRatio :: %.2f" % (len(test_u)/(len(test_u) + len(train_u))))

	return both_all, train_u, test_u

def count_trigrams(c, verbose=True):
	"""
	counts trigrams in a list of sentences
	we use this to find the trigrams in our gold sentences
	and then SUBTRACT that from the counts in our syntactic tagger
	so it's **like** we've never seen them ;)
	"""
	trigrams = {}
	total_count = 0
	if verbose:
		print("Counting trigrams...")
	for s in c:
		for i in range(2,len(s)):
			# probably a smarter way to do this rather than re-calc each iteration....
			t_older = split_tagged_lemma(s[i-2])[1]
			t_old = split_tagged_lemma(s[i-1])[1]
			t_cur = split_tagged_lemma(s[i])[1]
			this_trigram = (t_older, t_old, t_cur)
			total_count += 1
			if this_trigram in trigrams:
				trigrams[this_trigram] += 1
			else:
				trigrams[this_trigram] = 1

	if verbose:
		print("Total trigram types:", len(trigrams))
		print("\tTotal trigrams:", total_count)

	return trigrams

def score_tagger(gold_sentences, tagger, morph_weight = 1., syn_weight = 1., \
	verbose = True, max_guesses = 5):
	"""
	takes a list of sentences and compares their tags
	"""
	all_words = 0
	all_guesses = 0
	wrong_types = set([])
	all_types = set([])
	wrong = 0
	i = 0
	if verbose:
		print("Word\tGold\tGuess")
	for s in gold_sentences:
		i+=1
		# gotta split the words in the gold sentences since they all
		# already have tags on them.....
		gold = [split_tagged_lemma(w)[1] for w in s]
		s = [split_tagged_lemma(w)[0] for w in s]
		
		guess, g_info = tagger.tag(s, morph_weight = morph_weight, \
			syn_weight = syn_weight, mc = max_guesses)
		
		all_words += len(s)
		all_guesses += len(g_info)
		for t in range(len(gold)):
			all_types.add(s[t])
			# if our tag doesn't match the gold tag AND the gold tag is not N/A (EANC didn't know)
			# what the right tag was
			if gold[t] != guess[t] and gold[t] != "N/A":
				if verbose:
					print(s[t] + "\t" + gold[t] + "\t" + guess[t])
				wrong += 1
				wrong_types.add(s[t])
	if verbose:
		print()
	if True:
		print("Morphology Weight:", morph_weight, "\t\tSyntax Weight:", syn_weight)
		print(wrong, "out of", all_guesses, "wrong guesses. (%.3f)" % (1-(wrong/all_guesses)))
		#print(len(wrong_types), "out of", len(all_types), "wrong word types. (%.3f)" % (1-(len(wrong_types)/len(all_types))))
		print((all_words - wrong), "out of", all_words, "right. (%.3f)" % ((all_words - wrong)/all_words))

	# return the total accuracy
	return ((all_words - wrong)/all_words)