import re, sys, random

def separate_tokens(token_file):
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
    with open(token_file, "r") as rF:
        i = 0
        for line in rF:
            i+=1
            line = line.rstrip()
            l = line.rsplit("\t")
            if l[0] == "":
                continue
            token = l[0]
            try:
                tokens[token] = (l[1], l[2], l[3])
            except:
                tokens[token] = (l[1], [], [])
            # if there is no token, just skip skip skip
            tags = re.sub("[\['\]]", "", l[2]).rsplit(",")          
            tag_set = set([])
            for t in tags:
                if t != "":
                    tag_set.add(t)
            if len(tag_set) == 0:
                no_pos[token] = []
            elif len(tag_set) > 1:
                ambiguous[token] = list(tag_set)
            else:
                unique[token] = list(tag_set)
                
            # if we can't split it, it's because there's nothing, ie no POS
            
    return tokens, unique, ambiguous, no_pos

def read_hy_unigrams(unigram_file):
    """
    reads in a file of Unigrams from the Armenian wiki and returns a dictionary
    """
    u = {}
    with open(unigram_file, "r") as rF:
        for l in rF:
            l = l.rstrip().rsplit("\t")
            u[l[0]] = l[1]

    return u

