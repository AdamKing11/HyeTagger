import re, sys, nltk
from random import shuffle
from bclass_cross import *

from word_features import *

class mommaTagger:


	mommaTagger = 0
	accuray = 1.

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
		if verbose:
			print("\nNow training Naive Bayesian Classifier on tag trigrams.")
		for t in self.tag_trigrams:
			features = {}			# create new dict for the features
			
			# we find the ratio of this trigram to all other trigrams with
			# the same context. This way, we weight more frequent trigrams
			# higher #Bayes #prior
			
			# bin the ratios into 20%, 40%, 60%... etc
			features['previous'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[0]==t[0]])))

			features['following'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[2]==t[2]])))

			features['whole-context'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[0]==t[0] and c[2]==t[2]])))

			
			training_set.append((features, t[1]))	# we add this to the training set...

		if verbose:
			print("Training with", len(training_set), "data.")
		# randomize the order
		shuffle(training_set)

		self.mommaTagger = nltk.NaiveBayesClassifier.train(training_set)

	def test_momma_classifier(self, k, verbose = True):
		"""
		does a quick k-fold test on the classifier we've built... or at least on
		the features/data we've used to make the classifier....
		"""
		training_set = []
		
		for t in self.tag_trigrams:
			features = {}			# create new dict for the features
			
			features['previous'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[0]==t[0]])))

			features['following'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[2]==t[2]])))

			features['whole-context'] = round(5 * (self.tag_trigrams[t] \
				/ sum([self.tag_trigrams[c] for c in self.tag_trigrams \
				if c[0]==t[0] and c[2]==t[2]])))

			training_set.append((features, t[1]))	# we add this to the training set...


		# shuffle the order of the list
		shuffle(training_set)
		self.accuracy, _ = cross_validation(training_set, k, verbose = verbose)



if __name__ == "__main__":
	momma = mommaTagger("tagged.txt")
	momma.test_momma_classifier(6)
	for t in sorted(momma.tag_trigrams, reverse=True):
		l = sum([momma.tag_trigrams[x] for x in momma.tag_trigrams if x[0]==t[0] and x[2]==t[2]])
		#print(t, momma.tag_trigrams[t], l)
	momma.mommaTagger.show_most_informative_features()
		
	#print(len(momma.tag_trigrams))

