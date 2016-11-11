import sys, re

"""
Easy and dumb. We take a file with 1 sentence per line and add
a symbol for sentence start and sentence end
"""

if __name__ == "__main__":

    rF = open(sys.argv[1], "r")
    wF = open(sys.argv[2], "w")

    for line in rF:
        line = line.rstrip()
        # also, let's get rid of those $1, $2, $3 things that show up in the
        # wiki files... I suspect that those have to do with edit notes, not
        # actual text
        if re.search("\$[0-9]", line):
        	continue
        wF.write("<s> " + line + " </s>\n")

    rF.close()
    wF.close()
