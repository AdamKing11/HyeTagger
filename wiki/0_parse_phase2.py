import re, sys

def armChar(c):
	"""
	takes character as argument and returns True if the character is an
	Armenian letter, punctuation
	"""
	if len(c) == 1 and ord(c) >= 1328 and ord(c) < 1424:
		return True
	return False

def armLine(l):
    """
    returns true if more than half of the non-whitespace characters are Armenian
    """
    l = re.sub(" ","",l)
    total_chars = len(l)
    num_arm_chars = 0
    for c in l:
        if armChar(c):
            num_arm_chars += 1
        if num_arm_chars >= total_chars/2:
            return True
    return False

def parse_phase_2():
    """
    Meant to be run AFTER the initial clean-up of the Armenian wiki dump
    (the one in the shell script)
    This stage goes through and removes all lines that don't have 50% or 
    more Armenian characters (ignoring space)

    This way, we get rid of most of the Russian, English, random other stuff, etc
    """
    i = 0
    j = 0
    try:
        rF = open(sys.argv[1], "r")
        wF = open(sys.argv[2], "w")
        bF = open("bad."+sys.argv[2], "w")
    except:
        return False
    
    for line in rF:
        print("Doing line", i, end="\r")
        line = line.rstrip()
        if armLine(line):
            i += 1
            wF.write(line + "\n")
            pass
        else:
            j += 1
            bF.write(line + "\n")
            pass    
    print(i, "lines written to", sys.argv[2] + ",", j,"lines withheld.")

if __name__ == "__main__":
    parse_phase_2()
