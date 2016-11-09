import re, sys, nltk, random, numpy

# for loading in files...
from token_reader import *
from word_features import *


def read_hy_corpus(hyWiki_file, sentences = 10, min_length = 5):
    """
    super simple, read in the 'hyWiki_file' and return a list of lists for each
    sentence
    """
    hySentences = []
    # open the file
    # here, we're reading in the sentences from the hyWiki and putting them in
    # a list to try tagging....
    with open(hyWiki_file, "r") as rF:
        i = 0
        for line in rF:
            line = line.rstrip().rsplit(" ")
            # make sure the sentence is long enough
            if len(line) < min_length:
                continue
            i += 1
            hySentences.append(line)
        # if we have max sentences as 0, just do all
            if i >= sentences and sentences != 0:
                break
    return hySentences

def train_pos_classifier(unambig_tokens, tag_list = None):
    """
    We load in the EANC token file and build a classifier for UNAMBIGUOUS tokens
    
    As an extra, we can specify WHICH tags we want to look at, i.e. make a
    classifier for ONLY N v Adj or whatever
    """
    # create the training 
    training_set = []     
   
    # put the unambiguous tokens into the right format for the classifier
    unique_tokens = [(token, split_tagged_lemma(unambig_tokens[token][0])) \
            for token in unambig_tokens]

    all_tags = set([])
    for ut in unique_tokens:
        # get the token and lemma from the EANC database        
        token = ut[0]
        lemma = ut[1][0]
        tag = ut[1][1]
        # if we were passed a list of tags AND this tag is not in it, SKIP
        if tag_list != None and tag not in tag_list:
            continue
        # otherwise, we just add the features (as long as it's a good tag)
            # for some reason, some BAD tags exist in EANC and they all are in
            # all lower so the upper() part sorts that out....
        features = addFeatures(token, lemma)
        if tag == tag.upper():
            training_set.append((features,tag))
            all_tags.add(tag)

    return nltk.NaiveBayesClassifier.train(training_set)


###############################################################################

def stoopid_tag(hySentences, general_tagger, unambig, ambig):
    """
        hySentences - list of list of words to parse
        general_tagger - Bayesian classifier for ALL tags
        unambig - set of ambiguous tokens
        ambig - set of AMBIGUOUS tokens

    """
    i= -1
    # cycle through sentences
    tSentences = []
    for s in hySentences:
        tSentences.append([])
        i += 1        
        print("\tDoing sentence:", i+1, end="\r")
        j = -1
        # cycle through words
        for w in s:
            # skip the blank words... if they come up
            if w == "":
                continue
            j += 1
            # just fold case right now... maybe later we work in case-sensitive
            w = w.lower()
        #
        # UNAMBIGUOUS WORD
        #   
            tagged_entry = ()
            if w in unambig:
                tagged_entry = (1, w, (1, split_tagged_lemma(unambig[w][0])[1]))
                
        #####################
        # AMBIGUOUS WORD
        #
            elif w in ambig:
        #
                # for this, let's just assign equal probability to each possible
                # POS tag for the ambiguous labels.
                # (we'll let later syntactic stuff figure them out)
                
                # get all the tags from the various lemmas
                tags = set([split_tagged_lemma(lemma)[1] for lemma in ambig[w]])
                # put it in the right format for our classifier dictionary
                tags = tuple(sorted(tags))
                # loop through the lemmas and find the best
                k = 0
                guess = []
                for t in tags:
                    guess.append(1./len(tags))
                    guess.append(t)
                guess = tuple(guess)
                # b/c sometimes tokens with amibiguous lemmas have lemmas of the
                # same POS, we can just call them unambiguous...
                if len(guess) < 3:
                    tagged_entry = (1, w, (1, guess[1]))
                else:
                    tagged_entry = (2, w, guess)

        ####################    
        # UNKNOWN WORD        
            else:
        #        
        #
                # we don't know the word, so we'll have to use the general 
                # classifier
                # assume that the lemma = token
                guess_matrix = [(0,0) for x in range(3)]
                guess = general_tagger.prob_classify(addFeatures(w,w))
                k = 0
                for g in sorted(guess.samples(), key=guess.prob, \
                    reverse=True)[:3]:
                    guess_matrix[k] = (guess.prob(g), g)
                    k += 1          
                # we'll save the best 3 results   
                guess_matrix = sorted(guess_matrix, reverse=True)[:3]
                guess_matrix = tuple([x for y in guess_matrix for x in y])
                tagged_entry = (3, w, guess_matrix)
    
            # add the tagged entry to the list
            tSentences[i].append(tagged_entry)
    print("Done.")
    return tSentences                
       
if __name__ == "__main__":
    file_write_style = "w"
    print("Loading file and training... ")
    _, unambiguous, ambiguous, _ = separate_tokens("hyTokens.1-15.withpunc.txt")

    clsfier_all = train_pos_classifier(unambiguous)
    
    # read in 10 sentences
    hySentences = read_hy_corpus("hyWiki.final.1.txt", sentences = 0, \
        min_length = 10)
    print("Read in", len(hySentences), "sentences.")

    st = stoopid_tag(hySentences, clsfier_all, unambiguous, ambiguous)
    print("Done tagging... writing now.")
    
    i = 0
    golds = []
    silvers = []
    worst = []
    with open("parsed.hyWiki.txt", file_write_style) as wF:
        total_probs = []
        for ss in st:
            i+=1
            print("Writing sentence", i, end="\r")
            unambiguous_tagged = 0
            guess_prob = []
            for ww in ss:
                guess_prob.append(float(ww[2][0]))
                if ww[0] == 1:
                    unambiguous_tagged += 1 
                #################################
                # write the string
                writeString = str(ww[0]) + "\t" + ww[1]
                for wl in ww[2]:
                    writeString += "\t" + str(wl)
                wF.write(writeString + "\n")
                #################################
            total_probs.append(sum(guess_prob)/len(guess_prob))
            # if this is a realllly good one, add it to the gold list
            if total_probs[-1] == 1:
                golds.append(ss)
            # if it's pretty good but not perfect, silver
            elif total_probs[-1] > .95:
                silvers.append(ss)
            # let's also save the WORST sentence to see what it's like....
            if total_probs[-1] == min(total_probs):
                worst = ss           

            #print("Sentence", i, " -- ", unambiguous_tagged, "/", len(ss))
            #print("\tMean probability:", total_probs[-1])            
            wF.write(str(i) + "\t" + str(len(ss)) + "\t" + str(total_probs[-1]))
            wF.write("\n\n")
    print("\nSaved", len(st), "sentences.")
    print("Mean probability across all:", sum(total_probs)/len(total_probs))
    print("Min:", min(total_probs), "\tMax:", max(total_probs))
    print("Gold sentences:", len(golds))
    print("Silver sentences:", len(silvers))
    print()

    # write out the gold sentences
    with open("golds.parsed.txt", file_write_style) as wF:
        for gs in golds:
            for gw in gs:
                writeString = str(gw[0]) + "\t" + gw[1]
                for gl in gw[2]:
                    writeString += "\t" + str(gl)
                wF.write(writeString + "\n")
            wF.write("\n")
    
    # save the silver sentences
    with open("silver.parsed.txt", file_write_style) as wF:
        for ss in silvers:
            for sw in ss:
                writeString = str(sw[0]) + "\t" + sw[1]
                for sl in sw[2]:
                    writeString += "\t" + str(sl)
                wF.write(writeString + "\n")
            wF.write("\n")
        

    print("Worst sentence -", min(total_probs))
    for w in worst:
        print(w)          
