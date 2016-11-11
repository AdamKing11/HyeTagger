import re, sys

def split_tagged_lemma(tagged_lemma):
    """
    takes a tagged lemma - dog_N 
    and returns LEMMA, TAG - dog,N
    """
    # could do this on one line, easier to read this way
    lemma = re.sub("\_.*$","",tagged_lemma) 
    tag = re.sub("^.*\_","", tagged_lemma)
    return lemma, tag 


def armConst(c):
    """
    returns true if it is an Armenian consonant
    """    
    return bool(re.search(\
    "[ԲԳԴԶԹԺԼԽԾԿՀՁՂՃՄՅՆՇՉՊՋՌՍՎՏՐՑՒՓՔՖ]", c.upper()))

def armVowel(c):
    """
    returns true if it is an Armenian vowel
    """
    return bool(re.search("[ԱԵԸԻՈՒՕ]", c.upper()))


def addFeatures(token, lemma):
    """
    takes an Armenian word and returns a DICT of features and values for those
    feature for our eventual classifier
    """
    feat = {}
    lw = token.lower()
    bw = token.upper()
    w = token
    
    # check to see if word is capitalized (only first letter)
    feat['capitalized'] = bool(w[0] == bw[0] and not token == bw)

    # if theres a number 0-9 in there, prob a number...
    feat['has-numeral'] = bool(re.search("[0-9]",lw))
    
    # ratio of numerals to other stuff, trying to make sure we catch numbers...
    feat['numeral-ratio'] = len(re.findall("[0-9]",lw))/len(lw)
    
    # if there is some punctuation in it, prob a punctuation mark...
    feat['has-punc'] = bool(re.search("[«»՞–—ՙ՚՛՜՝՟]",lw))

    # get length
    #feat['length'] = len(w)    
    
    # get first letters
    feat['first-1'] = lw[0]
    feat['first-2'] = lw[0:1]
 
    # find "root" (what token and lemma share) and return what comes after for
    # the token 
    for i in range(min([len(token),len(lemma)])):
        if lemma[i] != token[i]:
            break
    try:
        suffix = token[i:]
        root = token[:i]
        feat['suffix-1'] = suffix[-1]
    except:
        suffix = ''
        root = token

    feat['suffix'] = suffix
    feat['root'] = root

    # get final letters
    feat['final-1'] = lw[-1]
    feat['final-2'] = lw[-2:]
    feat['final-3'] = lw[-3:]
    feat['final-31'] = lw[-3:-1]
    feat['final-32'] = lw[-3:-2]    
    feat['final-4'] = lw[-4:]
    feat['final-41'] = lw[-4:-1]
    feat['final-42'] = lw[-4:-2]
    feat['final-43'] = lw[-4:-3]
    
    # might have some effect....
    feat['vowel-initial'] = armVowel(bw[0])

    # if final character is either ն or ը, it might be the nom/acc marking
    feat['nom-acc_case'] = bool(re.search("ը$",lw)) or (armConst(w[-2:-1]) and \
        bool(re.search("ն$", lw)))
    # if it ends with ից, it's probably ablitive
    feat['abl-case'] = bool(re.search("ից$", lw))

    # if it ends with one of the common genitive endings, probably a noun
    feat['gen-case'] = bool(re.search("(ոջ|ի)$", lw))
    
    # lots of past verbs have է in the last 2 or 2 characters
    feat['է-past'] = bool(re.search("է.?$",lw))

    # if the word has the plural suffix [նք]եր, prob a noun
    feat['plural-suf'] = bool(re.search("[նք]եր$",lw))

    # if the word has the infinitive suffix, probably a verb
    feat['inf-suf'] = bool(re.search("[եա]լ$", lw))

    # if the word has the verbal ած participlial suffix, prob a verb
    feat['verb-part-suf'] = bool(re.search("ած$", lw))

    # if word has the sequence եց or աց, prob a causitive verb
    feat['causitive'] = bool(re.search("[աե]ց",lw))
    
    return feat

def format_training_data(all_tokens):
    """
    we take in a list of tokens with their lemmas and create a list of the TAG and
    the features for the given token --- use this for training and testing the
    Naive Bayes classifiers on word-morphology-stuff

    we'll also count the types of tags we get and return that too, nbd
    """
    training_set = []
    tag_set = {}     
        # put the unambiguous tokens into the right format for the classifier
    
    # some tags we DON'T want to train on, because they're closed sets, ie
    # punctuation or START, END
    closed_tags = ("START", "END", "PUNC")
    for token in all_tokens:
            ##### for continuity, the unambigous dictionary stores LISTS
            # make sure we do the 0th element of the length 1 list
        lemma, tag = split_tagged_lemma(all_tokens[token][0]) 
        if tag in closed_tags:
            continue
        # for some reason, some BAD tags exist in EANC and they all are in
        # all lower so the upper() part sorts that out....
        features = addFeatures(token, lemma)
        if tag == tag.upper():
            training_set.append((features,tag))
            if tag in tag_set:
                tag_set[tag] += 1
            else:
                tag_set[tag] = 1
                
    return training_set, tag_set