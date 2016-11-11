import sys, re

"""
opens a file rF with cleaned, punctuation separated text and
puts one sentence per line
"""

if __name__ == "__main__":

    stop_chars = "[:։:] "

    rF = open(sys.argv[1], "r")
    wF = open(sys.argv[2], "w")

    for line in rF:
        line = re.sub(stop_chars, ":\n", line)
        wF.write(line)

    rF.close()
    wF.close()
