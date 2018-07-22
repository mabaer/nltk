import nltk 
import random
from nltk.corpus import movie_reviews
    
# classify calculates the accuracy with a NB classifier
#
# @param  train_set     Train set
# @param  test_set      Test set
# @return               Accuracy
#
# @author Marc-André Bär
def classify(trainSet, testSet): 
    classifier = nltk.NaiveBayesClassifier.train(trainSet) 
    # Print the five most informative features for improvements
    classifier.show_most_informative_features(5) 
    return nltk.classify.accuracy(classifier, testSet)

# getFeatures returns a list of features
#
# @param  data         Data
# @return              List of features
#
# @author Marc-André Bär
def getFeatures(data): 
    return [(getSet(words), category) for (words, category) in data]

# getSet returns the feature set
#
# @param  words         Words
# @return               Feature set
#
# @author Marc-André Bär
def getSet(words):
    # Define positive and negative words
    positiveWords = ["hilarious", "terrific", "wonderful", "good", "great", "best","excellent", " perfect", "brilliant", "well", "interesting", "greatest", "super", "better"] 
    negativeWords = ["idiotic", "stupidity", "ludicrous", "boring", "bad", "worse", "worst", "poor", "long", " terrible", "weak", "horrible", " ridiculous", "stupid"]
    features = {} 
    countPos = 0 
    countNeg = 0   
    for token in words.keys():
        # Interprete the occurence of the token
        if words[token] < 2:
            features[token] = "exist" 
        elif words[token] >= 2 and words[token] < 4: 
            features[token] = "sometimes" 
        else: 
            features[token] = "often"
        # Check if it is a positive or a negative word
        if token in positiveWords: 
            countPos += 1 
        if token in negativeWords: 
            countNeg += 1 
             
    if countPos >= countNeg: 
        features['guess'] = "positive" 
    elif countNeg > countPos: 
        features['guess'] = "negative"                        
    return features 
    
# prepare review data as a list of tuples: 
# (list of tokens, category) 
# category is positive / negative 
review_data = [(movie_reviews.words(fileid), category) 
    for category in movie_reviews.categories() 
    for fileid in movie_reviews.fileids(category)]

threshold = 10000 # 10000 appears to be the best threshold

fd_all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words()) 
top_words = [word for (word, freq) in fd_all_words.most_common(threshold)] 

review_data_fdist = [(nltk.FreqDist(token.lower() for token in words if token in top_words), category) 
    for words, category in review_data]
# Shuffle data randomly    
random.seed(42) 
random.shuffle(review_data_fdist) 
# Split in training and test set 
trainSize = int(0.8 * len(review_data_fdist)) 
trainData, testData = review_data_fdist[:trainSize], review_data_fdist[trainSize+1:] 
# Get the features
trainFeatures = getFeatures(trainData) 
testFeatures = getFeatures(testData)
# Get and print the accuracy
print("Accuracy = %s" % classify(trainFeatures, testFeatures)) 