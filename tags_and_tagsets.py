import re
import nltk, random
from nltk.corpus import brown, names
    
# get_tagged_words creates an alphabetically sorted list of the distinct words by a given tag
#
# @param  tag           Tag
# @param  wordlist      Wordlist
# @return               Alphabetically sorted list of the distinct words
#
# @author Marc-André Bär
def get_tagged_words(tag, wordlist):
    # Just use lower case words
    lowerword=' '.join(wordlist).lower()
    # Tokenize the list of words
    tokenList = nltk.word_tokenize(lowerword)
    # Get the tags of the words
    tags = nltk.pos_tag(tokenList)
    # Return a list of words with the given tag (without duplicates)
    return sorted(list(set([word for word,pos in tags if (pos == tag)])))
    
# get_nouns_third_person creates a list which contains words that can be plural nouns or third person singular verbs 
#
# @param  wordlist      Wordlist
# @return               Alphabetically sorted list of the distinct words
#
# @author Marc-André Bär
def get_nouns_third_person(wordlist):
    # Just use lower case words
    lowerword=' '.join(wordlist).lower()
    # Tokenize the list of words
    tokenList = nltk.word_tokenize(lowerword)
    # Get the tags of the words
    tags = nltk.pos_tag(tokenList)
    # Get a list of plural nouns and third person singular verbs
    nounlist = list(set([word for word,pos in tags if (pos == 'NNS')]))
    thirdpersonlist = list(set([word for word,pos in tags if (pos == 'VBZ')]))
    # Return the intersection of both lists
    return sorted([val for val in nounlist if val in thirdpersonlist])
    
# get_three_word_phrases creates an alphabetically sorted list of three-word prepositional phrases of the form ADP + DET + NOUN
#
# @param  wordlist      Wordlist
# @return               Alphabetically sorted list of three-word prepositional phrases
#
# @author Marc-André Bär
def get_three_word_phrases(wordlist):
    # Just use lower case words
    lowerword=' '.join(wordlist).lower()
    # Tokenize the list of words
    tokenList = nltk.word_tokenize(lowerword)
    # Get the tags of the words
    tags = nltk.pos_tag(tokenList)
    # Iterate over the list and search for three-word prepositional phrases of the form ADP + DET + NOUN
    returnlist = []
    for index,tag in enumerate(tags):
        # Catch IndexErrors
        try:
            if (tag[1] == 'IN' and tags[index+1][1] == 'DT' and tags[index+2][1] == 'NN'):
                returnlist.append(tag[0] + ' ' + tags[index+1][0] + ' ' + tags[index+2][0])
        except:
            print('End of list')
    # Return a sorted list of the results        
    return sorted(list(set(returnlist)))
    
# get_gender_ratio calculates the male/female ratio of a given wordlist
#
# @param  wordlist      Wordlist
# @return               String of male and femeal ratio
#
# @author Marc-André Bär
def get_gender_ratio(wordlist):
    # Just use lower case words
    lowerword=' '.join(wordlist).lower()
    # Tokenize the list of words
    tokenList = nltk.word_tokenize(lowerword)
    # Get the tags of the words
    tags = nltk.pos_tag(tokenList)
    # Get a list of pronouns
    pronounlist = sorted(list(set([word for word,pos in tags if (pos == 'PRP')]))) 
    # Create a classifier for male and female names from the names corpus
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    # Shuffle the names randomly
    random.shuffle(labeled_names)
    # Use feature extractor on names
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    # Create train and test set
    train_set, test_set = featuresets[500:], featuresets[:500]
    # Cretate a naive bayes classifier on the trainset
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    # Count the male and female names
    malecount = 0
    femalecount = 0
    for name in pronounlist:
        if classifier.classify(gender_features(name)) == 'male':
            malecount+= 1
        else:
            femalecount+= 1
    totalcount = malecount + femalecount
    return 'Male: ' +  str(malecount)+ '(' + str(100*malecount/totalcount) + '%) | Female: ' + str(femalecount) + '(' + str(100*femalecount/totalcount) + '%)'
 
# Feature for the gender specification is the last letter of a name 
def gender_features(word):
    return {'last_letter': word[-1]}    

# Code to test the function
print('Words with the tag MD:')
print(get_tagged_words('MD', brown.words()))
print('Words that can be plural nouns or third person singular verbs:')
print(get_nouns_third_person(brown.words()))
print('Three-word prepositional phrases of the form ADP + DET + NOUN :')
print(get_three_word_phrases(brown.words()))
print('Ratio of masculine to feminine pronouns:')
print(get_gender_ratio(brown.words()))
print('')

# Make a cond freqdist for all tagged words within the brown corpus 
wordtags = nltk.ConditionalFreqDist((w.lower(), t) for w, t in brown.tagged_words(tagset="universal"))
# Make a list which just contains the numbers of tags per word
countlist = [len(list(wordtags[tags])) for tags in wordtags]
print('1 tag:')
# Print the count for each number of tags
print(countlist.count(1))
print('2 tags:')
print(countlist.count(2))
print('3 tags:')
print(countlist.count(3))
print('4 tags:')
print(countlist.count(4))
print('5 tags:')
print(countlist.count(5))
print('6 tags:')
print(countlist.count(6))
print('7 tags:')
print(countlist.count(7))
print('8 tags:')
print(countlist.count(8))
print('9 tags:')
print(countlist.count(9))
print('10 tags:')
print(countlist.count(10))

# Get the word with 6 tags
word = [tags for tags in wordtags if len(list(wordtags[tags])) == 6]
print(word)
print(list(wordtags['down']))

downsentences = [sent for sent in brown.sents() if 'down' in sent]
sentences = [sentence for sentence in downsentences]
# Print all sentences for down
for sentence in sentences:
    lowerword=' '.join(sentence).lower()
    # Tokenize the list of words
    tokenList = nltk.word_tokenize(lowerword)
    # Get the tags of the words
    tags = nltk.pos_tag(tokenList)
    #print(' '.join(sentence))
    #print(tags)
    

