"""
A Demo for the Armenian POS tagger 
Adam King 2016

A POS tagger for Armenian that uses a trigram-trained HMM
for sequence probabilities and a Naive Bayesian classifier
for morphological classification

This file will run through 6 different steps of the tagger
and print their accuracy when trying to tag 200 sentence while
trained on 1000 sentences (+ other stuff [sometimes])

(Hye is the Armenian word for Armenian)
"""

from lib.general import *
from lib.test_eval import *
from lib.taggers import *

import pickle

def step_0(test):
	"""
	First baseline, this builds a classifier that can ONLY look up words
	It trains on the 1000 training sentences, tests on the 200 testing sentences
	
	If it sees a word in the testing that it saw in the training (all words are 
	unambiguous in the training sentences), it tags it.
	If it doesn't recognize the word, it randomly guesses
	"""

	# loads in the tokens from the training data
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))

	# builds our "baby" classifier
	# the Baby component of the classifier does most of the grunt work
	# it holds the lists of ambiguous and unambiguous tokens AND the classifier (trained)
	# on the unambiguos tokens
	# in this case, we don't use the BC, we just build it for completeness sake
	baby = babyTagger(gold_words, read_from_file = False, verbose = False)

	# builds our "momma" classifier - the HMM
	# this reads through the training data and builds and HMM based on tag sequences
	# in this case, we build it but never use it - we're just going to guess unknown words
	momma = mommaTagger("data/wiki/hyWiki.golds.txt", tagged_c_type = 2, verbose = False)

	# builds big daddy, the classifier that weights input from the morphological classifier 
	# (baby) and contextual classifier (momma) and makes a decision with respect to unknown 
	# words or amibiguous tokens
	daddy = daddyTagger(baby, momma)
	
	# this prints out stats of the various components of the daddy classifier
	daddy.say_hello()

	# this will open up the 'test' data (which is just a passed array of sentences) and try to
	# tag it with the daddy tagger. Because the morpholical weight and syntactic weight are 0,
	# it will randomly guess for unknown words
	score_tagger(test, daddy, morph_weight = 0, syn_weight = 0, 
		verbose=False)


def step_1(test):
	"""
	Now, we're going to implement the HMM and see how it does...
	"""
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))

	baby = babyTagger(gold_words, read_from_file = False, verbose = False)
	momma = mommaTagger("data/wiki/hyWiki.golds.txt", tagged_c_type = 2, verbose = False)	
	daddy = daddyTagger(baby, momma)
	
	daddy.say_hello()
	# like above, but since 'syn_weight' is not 0, it will take the advice of the momma tagger
	# into consideration when it sees unknown/ambiguous words
	score_tagger(test, daddy, morph_weight = 0, syn_weight = 1, 
		verbose=False)



def step_2(test):
	"""
	Now it's time for the Bayesian Classifier to shine!
	"""
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))

	baby = babyTagger(gold_words, read_from_file = False, verbose = False)
	momma = mommaTagger("data/wiki/hyWiki.golds.txt", tagged_c_type = 2, verbose = False)
	daddy = daddyTagger(baby, momma)
	
	daddy.say_hello()
	# like above, but now the Bayesian classifier has some say in the tag
	score_tagger(hand, daddy, morph_weight = 1, syn_weight = 1, 
		verbose=False)


def step_3(test):
	"""
	Going to go through ALL of wikipedia and find ALL trigrams of unambiguous
	tokens to build a bigger, faster, meaner HMM
	"""
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))

	baby = babyTagger(gold_words, read_from_file = False, verbose = False)
	# I've Julia-Child'd and pre-built the classifier with all the trigrams, it takes a big...
	# this loads it in
	momma = c_load("taggers/m_tagger.t")
	daddy = daddyTagger(baby, momma)
	
	daddy.say_hello()
	score_tagger(hand, daddy, morph_weight = 1, syn_weight = 1, 
		verbose=False)


def step_4(test):
	"""
	Now, we try the iterative learning approach
	"""
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))
	# load in the ENTIRE list of tokens from Wikipedia. As with before, big list
	# so here they are already condensed....
	all_words_wiki = pickle.load( open( "taggers/lexicon.p", "rb" ) )

	# this is the part that does the training
	# goes through the list of ALL words from wikipedia, trys to classify them and 
	# if they beat the threshold, we re-train assuming the classifier got it right

	# uncomment to try playing with different iterations/thresholds
	#baby = babyTagger(gold_words, read_from_file = False, verbose = False)
	#baby.semisupervised_training(all_words_wiki, threshold = .995, iterations=1)	
	#c_save(baby, "taggers/b_tagger.semi.t")

	baby = c_load("taggers/b_tagger.semi.t")
	momma = c_load("taggers/m_tagger.t")
	daddy = daddyTagger(baby, momma)
	
	daddy.say_hello()
	score_tagger(hand, daddy, morph_weight = 1, syn_weight = 1, 
		verbose=False)


def step_5(test):
	"""
	Now, we try giving the BabyTagger a little extra formula to make him big and strong
	Allows us to bootstrap the Bayesian Classifier with tokens OUTSIDE the original training data
	"""
	gold_words = get_words(read_corpus("data/wiki/hyWiki.golds.txt"))

	baby = c_load("taggers/b_tagger.t")

	# Start the BabyTagger with 500 random tokens
	baby.limit_and_rebuild(500,verbose=False)
	# now, add back in the tokens from the training data
	baby.remember(unambig_to_r = gold_words, verbose=False)

	momma = mommaTagger("data/wiki/hyWiki.golds.txt", tagged_c_type = 2,verbose=False)
	daddy = daddyTagger(baby, momma)

	daddy.say_hello()
	score_tagger(hand, daddy, morph_weight = 1, syn_weight = 1, 
		verbose=False)

	

if __name__ == "__main__":
	# load in the test data, the hand tagged 200 sentences from EANC
	hand = read_corpus("data/EANC.200.hand.txt")

	print("*" * 15 + "Step 0" + "*" * 15)
	step_0(hand)
	print()

	print("*" * 15 + "Step 1" + "*" * 15)
	step_1(hand)
	print()

	print("*" * 15 + "Step 2" + "*" * 15)
	step_2(hand)
	print()

	print("*" * 15 + "Step 3" + "*" * 15)
	step_3(hand)
	print()

	print("*" * 15 + "Step 4" + "*" * 15)
	step_4(hand)
	print()

	print("*" * 15 + "Step 5" + "*" * 15)
	step_5(hand)
	print()

	