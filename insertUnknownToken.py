import nltk
from nltk.corpus import nps_chat
    
# insertUnknown creates a list of word/tag pairs with a separate token for unknown words
#
# @param  wordlist      Word/tag list
# @param  mostFreq      Most frequent words from the training Word/tag list
# @return               word/tag list with seperate Tag for unknown words
#
# @author Marc-André Bär
def insertUnknown(wordlist, mostFreq): 
    unknownListSentences = [] 
    for s in wordlist:   
        unknownListWords = [] 
        for (w,t) in s:
            # If the tag is not a noun and if the word is not in the most frequent list insert the new "unknown" token. Otherwise insert the word tag pair as it is
            if (t != "NOUN") and (w not in mostFreq): 
                unknownListWords.append(('X', t))
            else:
                unknownListWords.append((w, t))
        # Create the correct sentence structure 
        unknownListSentences.append(unknownListWords)
    return unknownListSentences    
    
# Code to test the function
tagged_posts = nps_chat.tagged_posts(tagset='universal') 
nr_posts = len(tagged_posts)
# Put train and dev set together
train_posts = tagged_posts[:(nr_posts*9)//10] #90%
test_posts = tagged_posts[((nr_posts*9)//10):] #10%
# Get the distribution within the train set and select the n most frequent words 
trainDist = nltk.FreqDist([w for train_words in train_posts for (w, t) in train_words]) 
mostFreq = [w for (w, f) in trainDist.most_common(3800)] 
train_postsGold = insertUnknown(train_posts, mostFreq) 
test_postsGold = insertUnknown(test_posts, mostFreq)
# Use a combination of taggers 
tagger1 = nltk.DefaultTagger('NOUN')
tagger2 = nltk.UnigramTagger(train_postsGold,backoff=tagger1)
tagger3 = nltk.BigramTagger(train_postsGold,backoff=tagger2)
print(tagger3.evaluate(test_postsGold))

