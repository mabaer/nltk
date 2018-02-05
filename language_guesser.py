import math
import nltk
from nltk.probability import FreqDist
from nltk.corpus import udhr
from nltk import bigrams
    
# build_language_models takes a list of languages and a dictionary of words as arguments and returns a conditional frequency distribution 
# where the languages are the conditions and the values are the lower case characters found in words
#
# @param  languages     list of languages
# @param  words         dictionary of words
# @param  mode          mode can be normal, tokens, character_bigrams or token_bigram
# @return               conditional frequency distribution 
#
# @author Marc-André Bär
def build_language_models(languages, words, mode="normal"):
    fdist = {}
    for l in languages:
        if mode == "normal":
            chars = []
            for w in words[l]:
                chars+=[c.lower() for c in w]
            fdist[l] = FreqDist(chars)
        elif mode == "tokens":
            fdist[l] = FreqDist([w.lower() for w in words[l]])           
        elif mode == "character_bigrams":
            fdist[l] = FreqDist(bigrams(words[l]))    
        elif mode == "token_bigram":
            fdist[l] = FreqDist(bigrams([w.lower() for w in words[l]]))
    return fdist

# guess_language returns the most likely language for a given text
#
# @param  language_model_cfd    list of languages
# @param  text                  dictionary of words
# @param  mode                  mode can be normal, tokens, character_bigrams or token_bigram
# @return                       most likely language 
#
# @author Marc-André Bär    
def guess_language(language_model_cfd, text, mode="normal"):
    # Get the right format
    if mode == "normal":
        textVal = [t.lower() for t in text]
    elif mode == "tokens":
        textVal = [t.lower() for t in text.split()]
    elif mode == "character_bigrams":
        textVal = list(bigrams([t.lower() for t in text]))     
    elif mode == "token_bigram":
        textVal = list(bigrams([t.lower() for t in text.split()]))
    # Current maximum and language
    max = 0
    lang = ""
    # Calculate the score for each language
    for language in language_model_cfd.keys():
        score = 0
        # Sum up for each part within the input the frequency
        for t in textVal:
            score += language_model_cfd[language].freq(t)
        # If the score is higher than the current highest score it will be replaced with the current one
        if (max <= score):
            max = score
            lang = language
    return lang


languages = ['English', 'German_Deutsch', 'French_Francais']
# build the language models
# udhr contains the Universal Declaration of Human Rights in over 300 languages
language_base = dict((language, udhr.words(language + '-Latin1')) for language in languages)
language_model_cfd = build_language_models(languages, language_base)
# print the models for visual inspection 
for language in languages:
    for key in list(language_model_cfd[language].keys())[:10]:
        print(language, key, "->", language_model_cfd[language].freq(key))
        
text1 = "I don't want to go to the office today."
text2 = "Pouvez-vous s'il vous plaît me donner plus de temps." 
text3 = "Das ist ein sehr langes Beispiel, aber was solls?"
 
# guess the language by comparing the frequency distributions 
print('guess for english text is', guess_language(language_model_cfd, text1)) 
print('guess for french text is', guess_language(language_model_cfd, text2)) 
print('guess for german text is', guess_language(language_model_cfd, text3))
