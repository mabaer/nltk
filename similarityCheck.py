import math
from nltk.corpus import wordnet as wn
    
# similarityCheck takes a list of pairs and returns a list in order of decreasing similarity based on the path_similarity function of their synsets
#
# @param  languages     List of pairs
# @return               Sorted list in order of decreasing similarity
#
# @author Marc-André Bär
def similarityCheck(pairList):
    retList = []
    for p in pairList:
        syn_sets1 = wn.synsets(p[1])  
        syn_sets2 = wn.synsets(p[0]) 
        max = 0
        # Use the combination of every snyset and return the maximum score
        for syn1 in syn_sets1:
            for syn2 in syn_sets2:
                score = syn1.path_similarity(syn2)
                # If the combination of the two synsets results into a higher score replace the maximum score
                if (score != None and max < score):
                    max = score
                    
        retList.append((p, max))
    # Sort the list in reverse order and return it
    sortedList = sorted(retList, key=lambda x: x[1], reverse=True)        
    return sortedList
    
# spearmanRankCorrelation takes two lists containing the same elements and returns the spearman rank correlation
#
# @param  list1         first list
# @param  list2         second list
# @return               Spearman rank correlation
#
# @author Marc-André Bär
def spearmanRankCorrelation(list1, list2):
    d_2 = 0
    n = len(list1)
    # Check if both lists have the same length
    if n != len(list2):
        return None
    pos1 = 0    
    for p in list1:
        # Enumerate over the second list and return the position
        for pos2, item in enumerate(list2):
            if item == p:
                d_2 += math.pow(math.fabs(pos1-pos2) , 2)
        pos1 += 1
    p = 1 - ((6 * d_2) / (n*(math.pow(n, 2) - 1)))
    return p

# Code to test the function
pairList=[('car', 'automobile'), ('gem', 'jewel'), ('journey', 'voyage'), ('boy', 'lad'), ('coast', 'shore'), ('asylum', 'madhouse'), ('magician', 'wizard'), ('midday', 'noon'), \
('furnace', 'stove'), ('food', 'fruit'), ('bird', 'cock'), ('bird', 'crane'), ('tool', 'implement'), ('brother', 'monk'), ('lad', 'brother'), ('crane', 'implement'), \
('journey', 'car'), ('monk', 'oracle'), ('cemetery', 'woodland'), ('food', 'rooster'), ('coast', 'hill'), ('forest', 'graveyard'), ('shore', 'woodland'), ('monk', 'slave'), \
('coast', 'forest'), ('lad', 'wizard'), ('chord', 'smile'), ('glass', 'magician') , ('rooster', 'voyage'), ('noon', 'string')]

simList = similarityCheck(pairList)
print(simList)
# Remove the scores from the tuples
cleanSimList = [e[0] for e in simList]
SRC=spearmanRankCorrelation(pairList, cleanSimList)
print('Spearman rank correlation: '+str(SRC))

