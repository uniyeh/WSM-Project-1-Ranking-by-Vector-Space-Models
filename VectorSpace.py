from pprint import pprint
from Parser import Parser
import util
import jieba
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import nltk
from DocVector import DocVector

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

    #Store original documents
    documents = []

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
        self.documents = documents  # Store original documents
        self.docVector = DocVector()
        self.parser = Parser(langOpt)
        if(len(documents)>0):
            self.build(documents, langOpt)

    def build(self,documents,langOpt):
        """ Create the vector space for the passed document strings """
        if (langOpt == 'CN'):
            # Tokenize Chinese documents using jieba and join with spaces
            documents = [' '.join(jieba.cut(document)) for document in documents]

        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)

        # Build TF vectors
        tfVectors = [self.makeVector(document) for document in documents]

        # Apply weighting scheme
        if self.weightOpt == 'TFIDF':
            # Calculate IDF
            self.idfVector = self.computeIDF(tfVectors)
            # Apply TF-IDF weighting
            tfidfVectors = [self.applyTFIDF(tfVector) for tfVector in tfVectors]
            # Normalize vectors (L2 normalization)
            self.documentVectors = [self.normalizeVector(vec) for vec in tfidfVectors]
        else:  # RawTF
            self.documentVectors = tfVectors
    
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


    def computeIDF(self, tfVectors):
        """ Compute IDF (Inverse Document Frequency) for all terms
        IDF(t) = log(n/(k+1)) with smoothing to avoid division by zero
        """
        import numpy as np
        numDocs = len(tfVectors)
        idfVector = [0] * len(self.vectorKeywordIndex)

        # Count how many documents contain each term
        for termIdx in range(len(self.vectorKeywordIndex)):
            docsWithTerm = sum(1 for tfVector in tfVectors if tfVector[termIdx] > 0)
            # Use smoothing: log(n/k)
            if docsWithTerm == 0:
                idfVector[termIdx] = 0
            else:
                idfVector[termIdx] = np.log(numDocs / docsWithTerm)

        return idfVector

    def applyTFIDF(self, tfVector):
        """ Apply TF-IDF weighting to a TF vector using Raw TF """
        # Raw TF: TF(t,d) = f(t,d) (no normalization)
        # IDF(t) = log(n/k) where n = # docs, k = # docs with term t
        return [tf * idf for tf, idf in zip(tfVector, self.idfVector)]

    def normalizeVector(self, vector):
        """ Normalize vector to unit length (L2 normalization) """
        import numpy as np
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def makeVector(self, wordString):
        """ Build TF vector for a document """
        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            if word in self.vectorKeywordIndex:
                vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model (Raw TF)
        return vector

    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        if self.langOpt == 'CN':
            # Tokenize Chinese query with jieba
            query_text = ' '.join(jieba.cut(' '.join(termList)))
            query = self.makeVector(query_text)
        else:
            query = self.makeVector(" ".join(termList))

        # Apply TF-IDF weighting and normalization if needed
        if self.weightOpt == 'TFIDF' and hasattr(self, 'idfVector'):
            query = self.applyTFIDF(query)
            query = self.normalizeVector(query)

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
            self.queryVector = self.buildQueryVector(query)

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
    

    def createFeedbackQuery (self, topDoc):
        """ feedback relevant documents """
        docText = ""
        for idx, doc in enumerate(self.fileNames):
            if topDoc[0] == doc:
                # Get the original document text
                docText = self.documents[idx]

        # Tokenize and POS tag using NLTK
        tokens = nltk.word_tokenize(docText)
        pos_tags = nltk.pos_tag(tokens)

        # Extract nouns and verbs (NN* and VB* tags)
        NNVBList = [word for word, pos in pos_tags if pos.startswith('NN') or pos.startswith('VB')]

        # Store feedback terms for display
        self.feedbackTerms = NNVBList

        # Build query vector from filtered words
        filteredText = ' '.join(NNVBList)
        docQueryVector = self.buildQueryVector([filteredText])

        # Combine original query with document feedback (Rocchio algorithm)
        import numpy as np
        feedbackQuery = np.array(self.queryVector) + 0.5 * np.array(docQueryVector)
        return feedbackQuery.tolist()

    def getFeedbackQueryText(self):
        """ Get the feedback query terms as text for display """
        if hasattr(self, 'feedbackTerms'):
            return ' '.join(self.feedbackTerms[:20])  # Show first 20 terms
        return ""
    
    def feedbackRel (self, feedbackQuery):
        """ feedback relevant documents """
        scores = []
        for i, docVector in enumerate(self.documentVectors):
            score = []
            score = util.cosine(feedbackQuery, docVector)  # cosine similarity
            scores.append((self.fileNames[i], score))
        scores.sort(key = lambda x: x[1], reverse = True)  # Sort scores in descending order
        return scores[:10]  # Return top 10 scores


class RankEvaluation(VectorSpace):
    
    # for every query, store the MAP, MRR, and Recall
    rankList = [] # len should be query size and each element should be a list of tuples (docID, score), and size is 10
    isRelQueries = [] # len should be query size and each element should be a list of 1 or 0, and size is 10
    map = 0
    mrr = 0
    recall = 0
    queryNames = []
    relDict = {}
    queryRelDict = {}

    def __init__(self, documents, fileNames, weightOpt, similarityOpt, langOpt, queryNames, relDict):
        super().__init__(documents, fileNames, weightOpt, similarityOpt, langOpt)
        self.queryNames = queryNames
        self.relDict = relDict
        if relDict:
            self.queryRelDict = self.__buildQueryRel(relDict)

    def __buildQueryRel(self, relDict):
        return {query: len(relevantDocs) for query, relevantDocs in relDict.items()}

    def __checkRel(self, i, docID):
        # print(self.relDict[self.queryNames[i]], docID)
        return 1 if docID.replace("d", "") in self.relDict[self.queryNames[i]] else 0
    
    def __calculateAP(self, isRelList):
        """ calculate the mean average precision """
        precisionQuery = 0
        count = 0
        for i, isRel in enumerate(isRelList):
            if isRel:
                count += 1
                precisionQuery += count / (i + 1)
        apQuery = precisionQuery / count if count != 0 else 0
        return apQuery

    def __calculateRR(self, isRelList):
        """ calculate the mean reciprocal rank """
        for i, isRel in enumerate(isRelList):
            if isRel:
                return 1 / (i + 1)
        return 0

    def __calculateRecall(self, idx, isRelList):
        """ calculate the recall """
        numRel = util.total(isRelList)
        # print(self.queryNames[idx])
        # print(self.queryRelDict[self.queryNames[idx]], isRelList, numRel / self.queryRelDict[self.queryNames[idx]] if self.queryRelDict[self.queryNames[idx]] != 0 else 0)
        return numRel / self.queryRelDict[self.queryNames[idx]] if self.queryRelDict[self.queryNames[idx]] != 0 else 0
    
    def __calculateMetric(self, idx, isRelList):
        apScore = self.__calculateAP(isRelList)
        rrScore = self.__calculateRR(isRelList)   
        recallScore = self.__calculateRecall(idx, isRelList)
        return apScore, rrScore, recallScore

    def makeRelList(self, queries):
        for i, query in enumerate(queries): # self.queryNames[i] is the query ID
            scores = super().rankDocuments(query)
            self.rankList.append(scores)
            # make a list of 1 or 0 for each document in the top 10 e.g., [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ...query size]
            isRel = [self.__checkRel(i, doc) for doc, score in scores]
            self.isRelQueries.append(isRel)
        ## Validate Code for check __checkRel function
        # totalQueriesCount = 0
        # for queryIdx, rank in enumerate(self.rankList):
        #     count = 0
        #     for docID, score in rank:
        #         if docID.replace("d", "") in self.relDict[self.queryNames[queryIdx]]:
        #             count += 1
        #     print(count, sum(self.isRelQueries[queryIdx]))
        #     if count != sum(self.isRelQueries[queryIdx]):
        #         print("Error: count != sum")
        #     else:
        #         totalQueriesCount += 1
        # print("Total correct queries count: ", totalQueriesCount)

    def evaluate(self):
        """ evaluate the performance of the vector space model """
        apList = []
        rrList = []
        recallList = []
        for i, isRelList in enumerate(self.isRelQueries):
            apScore, rrScore, recallScore = self.__calculateMetric(i, isRelList)
            apList.append(apScore)
            rrList.append(rrScore)
            recallList.append(recallScore)
        self.map = util.average(apList)
        self.mrr = util.average(rrList)
        self.recall = util.average(recallList)
        print("--------------------------")
        print('%-11s' % "MRR@10: ", self.mrr)
        print('%-11s' % "MAP@10: ", self.map)
        print('%-11s' % "Recall@10: ", self.recall)
        print("--------------------------")