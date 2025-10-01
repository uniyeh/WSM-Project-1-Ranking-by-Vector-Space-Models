from pprint import pprint
from Parser import Parser
import util
import jieba
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex = []

    #Collection of query term vectors
    queryVectors = []

    #Collectiorn of IDF vecotors
    idfVectors = []

    #Tidies terms
    parser=None
    weightOpt=None
    similarityOpt=None
    langOpt=None

    def __init__(self, documents=[], fileNames=[], weightOpt='TFIDF', similarityOpt='cosine', langOpt='EN'):
        self.fileNames = fileNames
        self.weightOpt = weightOpt
        self.similarityOpt = similarityOpt
        self.langOpt = langOpt
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents, langOpt)

    def build(self,documents,langOpt):
        """ Create the vector space for the passed document strings """
        if (langOpt == 'CN'):
            # Tokenize Chinese documents using jieba and join with spaces# Tokenize Chinese documents using jieba and join with spaces
            documents = [' '.join(jieba.cut(document)) for document in documents]
           
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]
    
    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        if self.langOpt == 'CN':
            # Tokenize Chinese query with jieba
            query_text = ' '.join(jieba.cut(' '.join(termList)))
            query = self.makeVector(query_text)
        else:
            query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings

    def rankDocuments(self, query):
        """ rank the documents in the document vector space according to their similarity to the query """
        scores = []
        # validated data list from TA
        # TANewList = ["News561", "News13136", "News5680", "News12524", "News9700", "New13100", "News5668", "News13924", "News6486", "News11212"]

        if self.langOpt == 'EN':
            self.queryVector = self.buildQueryVector(query)
        else:
            self.queryVector = self.buildQueryVectorCN(query)

        for i, docVector in enumerate(self.documentVectors):
            score = []
            if self.similarityOpt == 'cosine':
                score = util.cosine(self.queryVector, docVector)  # cosine similarity
            elif self.similarityOpt == 'Euclidean':
                score = util.Euclidean(self.queryVector, docVector)  # Euclidean distance
            else:
                raise ValueError("Invalid similarity measure.")
            scores.append((self.fileNames[i], score))
            # validate code
            # if self.similarityOpt == 'Euclidean' and self.fileNames[i] in TANewList:
            #     print(self.fileNames[i], score)

        # Sort documents by score in descending order and take top 10
        if self.similarityOpt == 'Euclidean':
            scores.sort(key = lambda x: x[1])
        else:
            scores.sort(key = lambda x: x[1], reverse = True)  # Sort scores in descending order
        return scores[:10]  # Return top 10 scores

if __name__ == '__main__':
    #test data
    documents = ["The cat in the hat disabled",
                 "A cat is a fine pet ponies.",
                 "Dogs and cats make good pets.",
                 "I haven't got a hat."]

    vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)

    #print(vectorSpace.documentVectors)

    print(vectorSpace.related(1))

    #print(vectorSpace.search(["cat"]))

###################################################
