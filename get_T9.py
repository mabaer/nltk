import re
import nltk
from nltk.probability import FreqDist
from nltk.corpus import webtext, nps_chat
    
# get_T9_word takes a digit sequences and works similar to the T9 system on mobile phones
#
# @param  digits        Digit sequences
# @return               Most common word
#
# @author Marc-André Bär
def get_T9_word(digits):
    blocks = ['[abc]', '[def]', '[ghi]', '[jkl]', '[mno]', '[pqrs]', '[tuv]', '[wxyz]']
    
    # Init the corpus (For performance reasons this should be done once outside of the function
    all_words = nltk.corpus.webtext.words() + nltk.corpus.nps_chat.words()
    # Convert all words to lower case
    all_words_lower = [w.lower() for w in all_words]

    # Get the regex string for the input
    regexStr = ''
    for d in map(int, digits):
        # Ignore zeros and ones
        if d == 0 or d == 1:
            continue
        regexStr += blocks[d-2]
        
    # Find all possible words
    possible_words = [w for w in all_words_lower if re.search('^'+ regexStr  +'$', w)]
    
    # Create a frequency distribution over the possible words
    freqDist = FreqDist(possible_words)

    # Return the most common possible word
    return freqDist.most_common(1)[0][0]


# Code to test the function
result = ''
for w in ['43556','73837','4','26','3463']: 
    result += get_T9_word(w) + ' '
print(result) 
