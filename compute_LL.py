import math
import nltk
from nltk.corpus import reuters, gutenberg
from nltk.probability import FreqDist

# compute_LL calculates the log likelihood score for a given phrase of a text and a given background corpus
#
# @param  phrase        given phrase
# @param  fdist_fg      base document
# @param  fdist_bg      background corpus
# @return               log likelihood score for the given phrase
#
# @author Marc-André Bär
def compute_LL(phrase, fdist_fg, fdist_bg):
    # Compute A,B,C,D for the word phrase in fdist_fg and the fdist_bg corpus as background
    A_phrase = fdist_fg[phrase]
    B_phrase = fdist_bg[phrase] 
    C = sum(fdist_fg.values()) # Size of the document
    D = sum(fdist_bg.values()) # Size of the corpus
    # Compute expectations
    E1_phrase = C * (A_phrase+B_phrase) / (C + D) # Expectation for the phrase in fdist_fg
    E2_phrase = D * (A_phrase+B_phrase) / (C + D) # Expectation for the phrase in fdist_bg
    # Compute LL for the phrase in fdist_fg and check for zero values
    if A_phrase == 0 and B_phrase == 0:
        LL_phrase = 0
    elif A_phrase != 0 and B_phrase == 0:
        LL_phrase = 2*(A_phrase * math.log(A_phrase/E1_phrase))
    elif A_phrase == 0 and B_phrase != 0:
        LL_phrase = 2*(B_phrase * math.log(B_phrase/E2_phrase))
    else:
        LL_phrase = 2*(A_phrase * math.log(A_phrase/E1_phrase) + B_phrase * math.log(B_phrase/E2_phrase))
    return LL_phrase

# Example code to call the function
reuters_words = reuters.words()
bg_corpus = nltk.bigrams(reuters_words)
# Get the bigrams of the whole corpus as freq dist 
fdist_bg = FreqDist(bg_corpus)
# Get the bigrams of the selected text as freq dist 
fdist_fg = FreqDist(nltk.bigrams(gutenberg.words('carroll-alice.txt')))
# Calculate the bigrams
bigramList = [(b, compute_LL(b, fdist_fg, fdist_bg)) for b in fdist_fg]
#Put the top ten of the bigrams in a list and print them
topTenSortedBigramList = sorted(bigramList, key=lambda x: x[-1], reverse=True)[:10]
print(topTenSortedBigramList)