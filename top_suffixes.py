import nltk
from nltk import *

# top_suffixes takes a sequence of words as an input and returns the 10 most frequent two-character sufﬁxes
# The two-character sufﬁx are the last two characters of any word of length 5 or more
#
# @param  inputList     a sequence of words as an input
# @return               the 10 most frequent two-character sufﬁxes
#
# @author Marc-André Bär
def top_suffixes(inputList):
    # Get last two chars from each word with more than four chars within the input list and put them into a list of suffixes
    subInputList = [w[-2:] for w in inputList if len(w) > 4]
    # Get the most common words suffixes from the list
    mostCommon = FreqDist(subInputList).most_common(10)
    # Make a list with suffixes out of the tuples
    mostCommon = [list(s)[0] for s in mostCommon]
    # Return the result
    return mostCommon

# Example code to call the function
words = nltk.corpus.gutenberg.words('austen-emma.txt') 
print(top_suffixes(words))