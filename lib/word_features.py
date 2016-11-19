import re, sys, pickle

def read_EANC(token_file, verbose = True, fc = True):
        """
    Reads in a list of tokens we've culled from EANC and returns a list of:
    TOKEN[token] - (unigram_count, TAGS, DEFS)
    UNIQUE[token] - [TAG]
    AMBIGUOUS[token] - [TAGS]
    NO_POS[token] - []  
        """
        all_tags = {}
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
                    print("\tReading token -", i, end="\r")
                line = line.rstrip()
                l = line.rsplit("\t")
                # if it's a blank line for whatever reason, SKIP
                if l[0] == "":
                    continue
                # the token in question is first thing on line... duh
                if fc:
                    token = l[0].lower()
                else:
                    token = l[0]
                try:
                    tokens[token] = (l[1], l[2], l[3])
                except:
                    # if we can't split it, it's because there's nothing, ie no POS
                    tokens[token] = (l[1], [], [])
                # if there is no token, just skip skip skip
                lemmas = re.sub("[\['\]]", "", l[2]).rsplit(",") 
                # here, we get all the tags for the various lemmas associated with the
                # token         
                lemma_set = set([])
                tag_set = set([])
                for l in lemmas:
                    if l != "":
                        # NOTE! we're not adding the tags, we're adding TAGGED lemmas to
                        # the list for each word
                        lemma_set.add(l)
                        # now, we get a list of tags
                        tag = split_tagged_lemma(l)[1]
                        tag_set.add(tag)
                        if tag in all_tags:
                            all_tags[tag] += 1
                        else:
                            all_tags[tag] = 1
                # now that we've gotten all the tags from the various lemmas,
                # decide which group to put the token in
                if len(lemma_set) == 0:
                    no_pos[token] = []
                elif len(lemma_set) > 1:
                    ambiguous[token] = list(lemma_set)
                else:
                    unique[token] = list(lemma_set)

        if verbose:
            print("\nTotal tokens:", len(tokens), "\tTotal tag types:", len(all_tags))
            print("\tUnambiguous:",len(unique))
            print("\tAmbiguous:",len(ambiguous))
            print("\tNo Label:",len(no_pos))

        # return all lists
        return unique, ambiguous, no_pos, tokens, all_tags


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
    return bool(re.search("[ԲԳԴԶԹԺԼԽԾԿՀՁՂՃՄՅՆՇՉՊՋՌՍՎՏՐՑՒՓՔՖ]", c.upper()))

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
    try:
        feat['capitalized'] = bool(w[0] == bw[0] and not token == bw)
    except:
        # sometimes, we get weird things passed in that mess up the upper
        # casing...
        pass

    # if theres a number 0-9 in there, prob a number...
    feat['has-numeral'] = bool(re.search("[0-9]",w))
    
    # ratio of numerals to other stuff, trying to make sure we catch numbers...
    try:
        feat['numeral-ratio'] = len(re.findall("[0-9]",w))/len(w)
    except:
        feat['numeral-ratio'] = 0
        
    # if there is some punctuation in it, prob a punctuation mark...
    feat['has-punc'] = bool(re.search("[«»՞–—ՙ՚՛՜՝՟]",w))

    # get length
    feat['length'] = len(w)    
    
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

    # չ
    feat['initial-neg'] = bool(re.search("^չ", lw))

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
    closed_tags = ("START", "END", "PUNC", "CONJ", "N/A", "INTJ")
    for token in all_tokens:
            ##### for continuity, the unambigous dictionary stores LISTS
            # make sure we do the 0th element of the length 1 list
        lemma, tag = split_tagged_lemma(all_tokens[token][0]) 
        if tag in closed_tags:
            continue
        # we don't want to try and classify a word that doesn't exist, do we?
        if len(token) == 0 or len(lemma) == 0:
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

def c_save(t, tagger_file):
    """
    saves classifier via pickle
    """
    pickle.dump(t, open(tagger_file, "wb" ))

def c_load(tagger_file):
    """
    loads a classifier
    """
    return pickle.load(open(tagger_file,'rb'))

def shrink_token_dict(unambig_big, tag_max = 50, tag_types = None):
    """
    take in a list of unambiguous tokens and REDUCE the dictionary
    to only be 'tag_max' of each tags
    ie, we come in with 8000 nouns, 7500 verbs...., return dict with only
    50 of each

    only reduce tags in 'tag_types', unless it's None, then do all
    """
    tag_count = {}
    u_little = {}

    for l in unambig_big:
        # get the tag for the token's lemma
        tag = split_tagged_lemma(unambig_big[l][0])[1]
        
        # if we were passed a list of tags to look at AND this tag is
        # not in that list, just add it to the returning list and continue
        if tag_types != None and tag not in tag_types:
            u_little[l] = unambig_big[l]
            continue

        if tag in tag_count:
            tag_count[tag] += 1
            if tag_count[tag] > tag_max:
                continue
        else:
            tag_count[tag] = 1

        u_little[l] = unambig_big[l]
    
    return u_little