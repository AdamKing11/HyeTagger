import re, sys, nltk
from random import shuffle
from bclass_cross import *

from word_features import *

class mommaTagger:


	mommaTagger = 0
	tag_trigrams = {} # we hold in counts for each tag trigram

	def __init__(self, tagged_c, verbose = True):
		"""

		"""
		self.read_tag_trigrams(tagged_c)
		self.train_momma_classifier()

	def read_tag_trigrams(self, baby_tagged_file, threshold = 1., verbose = True):
		"""
		read in a file tagged by the baby tagger and get all tag
		trigrams where ALL tokens in the trigram beat our threshold
		(default 1, so we only do unambiguous tokens)
		"""
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
				else:
					# however, if the score is too low, we re-start the list 
					# and continue
					tag_history = []
					continue
				if len(tag_history) == 3:
					j += 1
					tag_trigram = tuple([t[0] for t in tag_history])
					if tag_trigram in self.tag_trigrams:
						self.tag_trigrams[tag_trigram] += 1
					else:
						self.tag_trigrams[tag_trigram] = 1
					# now, we pop the OLDEST guy off the list
					tag_history.pop(0)
		if verbose:
			print("\nRead in", i-1, "sentences and found", j, "trigrams with", \
				len(self.tag_trigrams), "trigram types.")

	def train_momma_classifier(self, verbose = True):
		"""
		takes a bunch of tag trigrams and builds a Naive Bayesian train_momma_classifier
		that will judge a particular POS based on the preceding and following tag
		"""

		training_set = []

		for t in self.tag_trigrams:
			features = {}			# create new dict for the features
			features['previous'] = t[0]	# set the previous
			features['following'] = t[2]	# set the following
			training_set.append((features, t[1]))	# we add this to the training set...

		# randomize the order
		shuffle(training_set)

		self.mommaTagger = nltk.NaiveBayesClassifier.train(training_set)



if __name__ == "__main__":
	momma = mommaTagger("tagged.txt")
	momma.mommaTagger.show_most_informative_features()
	#for t in momma.tag_trigrams:
	#	print(t, momma.tag_trigrams[t])
	#print(len(momma.tag_trigrams))
