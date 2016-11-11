import re, sys

def separate_punc_outside(word, left_punc = None, right_punc = None):
    """
    takes a line of Armenian text and seperates the punctuation from the words
    « » ՞ – — ՙ ՚ ՛ ՜ ՝ ՟ ։ : , -
    [«»՞–—ՙ՚՛՜՝՟։:,-] –—ՙ՚՛՜՝՟՞
    """
    if left_punc == None:
        left_punc = "«\"(\\-"
    if right_punc == None:
        right_punc = "\",;»։:)\\-"
        
    # define the regexes
    left_reg = "^([" + left_punc + "])(.+)$"
    right_reg = "^(.+)([" + right_punc + "])$"
    
    # split left
    r = re.findall(left_reg, word)
    if len(r) > 0:
        word = r[0][0] + " " +separate_punc_outside(r[0][1])
    # split right
    r= re.findall(right_reg, word)   # p for prefix
    if len(r) > 0:
        word = separate_punc_outside(r[0][0]) + " " + r[0][1]
    
    # split inside punctuation
    
    return word

def separate_punc_inside(word, mid_punc = None):
    """

    """
    # now, we split up punctuation that comes OUTSIDE of the word
    # some armenian punctuation marks show up INSIDE the word, following 
    # stressed vowel, we'll put them behind the words they show up in
    if mid_punc == None:    
        mid_punc = "՞ՙ՚՛՜՝՟"
    # define the regEx
    mid_reg = "(.* )*(.*)([" + mid_punc + "])(.*)"

    # look for the punctuation...
    r = re.findall(mid_reg, word)
    if len(r)>0:
        # if we find it, put it in front of the word chunk
        word = r[0][0] + " " + r[0][2] + " " + r[0][1] + r[0][3]
    return word

def separate_punc(word):
    """
    """
    # first look for OUTSIDE punc
    word = separate_punc_outside(word)
    # then look for inside punc
    word = separate_punc_inside(word)

    # get rid of extra spaces
    word = re.sub("^ ", "", word)
    return re.sub("  ", " ", word)


def parse_punc_line(line):
    """
    takes a line of Armenian text and parses out the punctuation    
    """
    line = line.rstrip().rsplit(" ")
    parsed_line = ""
    for w in line:
        w = separate_punc(w)
        parsed_line += w + " "
    
    return parsed_line[0:-1]
    
################################################################################

if __name__ ==  "__main__":
    #rF = open(sys.argv[1], "r")
    #wF = open(sys.argv[2], "w")
    rF = open("hyWiki.prePUNC", "r")
    wF = open("hyWiki.punc_sep", "w")
    
    i = 0    
    for line in rF:
        i+=1
        print("Doing line", i, end= "\r")
        wF.write(parse_punc_line(line) + "\n")
    print()
    wF.close()
    rF.close()
    
