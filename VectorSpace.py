from pprint import pprint
from Parser import Parser
import util
import jieba
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import nltk
from DocVector import DocVector
import numpy as np

class Document:
    """Simple document wrapper for storing tokenized documents"""
    def __init__(self, text, words):
        self.text = text
        self.words = words
        self.tags = None  # Will be set by POS tagging if needed

class VectorSpace:
    """A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex = []

    #Tidies terms
    parser = None
    weightOpt = None
    similarityOpt = None
    langOpt = None

    def __init__(self, documents=[], fileNames=[], weightOpt='TFIDF', similarityOpt='cosine', langOpt='EN'):
        self.fileNames = fileNames
        self.weightOpt = weightOpt
        self.similarityOpt = similarityOpt
        self.langOpt = langOpt
        self.parser = Parser(langOpt)
        self.docVector = DocVector()
        self.documents = documents
        self.documentVectors = []

        if documents and self.langOpt == 'EN':
            self.__build(documents)
        elif documents and self.langOpt == 'CN':
            self.__buildCN(documents)
        else:
            if documents:
                raise ValueError("Invalid language option.")

    def __build(self, documents):
        """Create the vector space for English documents"""
        tokenizedDocs = [self.parser.tokenise(doc) for doc in documents]
        filteredDocs = [self.parser.removeStopWords(tokens) for tokens in tokenizedDocs]
        self.processedDocs = [Document(' '.join(words), words) for words in filteredDocs]
        self.vectorKeywordIndex = self.__getVectorKeywordIndex(self.processedDocs)
        self.documentVectors = self.__makeVector(self.processedDocs)

    def __buildCN(self, documents):
        """Create the vector space for Chinese documents using Jieba"""
        # Use Jieba for Chinese word segmentation
        segmentedDocs = [list(jieba.cut(doc)) for doc in documents]
        self.processedDocs = [Document(' '.join(words), words) for words in segmentedDocs]
        self.vectorKeywordIndex = self.__getVectorKeywordIndex(self.processedDocs)
        self.documentVectors = self.__makeVector(self.processedDocs)

    def __getVectorKeywordIndex(self, processedDocs):
        """Create the keyword associated to the position of the elements within the document vectors"""
        vocabularyString = [word for doc in processedDocs for word in doc.words]
        uniqueVocabularyList = util.removeDuplicates(vocabularyString)
        return {word: idx for idx, word in enumerate(uniqueVocabularyList)}

    def __makeVector(self, processedDocs):
        """Build document vectors with TF-IDF weighting"""
        documentVectors = []
        self.idfVector = self.docVector.buildIdfVector(processedDocs, self.vectorKeywordIndex)
        for doc in processedDocs:
            tfVector = self.docVector.buildTfVector(doc, self.vectorKeywordIndex, self.weightOpt)
            tfidfVector = self.docVector.buildIfidfVector(tfVector, self.idfVector, self.weightOpt)
            documentVectors.append(tfidfVector)
        return documentVectors

    def buildQueryVector(self, query):
        """Convert English query string into a term vector"""
        tokens = self.parser.tokenise(query)
        filteredTokens = self.parser.removeStopWords(tokens)
        queryDoc = Document(' '.join(filteredTokens), filteredTokens)
        queryTfVector = self.docVector.buildTfVector(queryDoc, self.vectorKeywordIndex, self.weightOpt)
        queryVector = self.docVector.buildIfidfVector(queryTfVector, self.idfVector, self.weightOpt)
        return queryVector

    def buildQueryVectorCN(self, query):
        """Convert Chinese query string into a term vector"""
        words = list(jieba.cut(query))
        queryDoc = Document(' '.join(words), words)
        queryTfVector = self.docVector.buildTfVector(queryDoc, self.vectorKeywordIndex, self.weightOpt)
        queryVector = self.docVector.buildIfidfVector(queryTfVector, self.idfVector, self.weightOpt)
        return queryVector

    def rankDocuments(self, query):
        """Rank the documents in the document vector space according to their similarity to the query"""
        scores = []

        if self.langOpt == 'EN':
            self.queryVector = self.buildQueryVector(query)
        else:
            self.queryVector = self.buildQueryVectorCN(query)

        for i, docVector in enumerate(self.documentVectors):
            if self.similarityOpt == 'cosine':
                score = util.cosine(self.queryVector, docVector)
            elif self.similarityOpt == 'Euclidean':
                score = util.Euclidean(self.queryVector, docVector)
            else:
                raise ValueError("Invalid similarity measure.")
            scores.append((self.fileNames[i], score))

        # Sort documents by score
        if self.similarityOpt == 'Euclidean':
            scores.sort(key=lambda x: x[1])  # Ascending for distance
        else:
            scores.sort(key=lambda x: x[1], reverse=True)  # Descending for similarity
        return scores[:10]

    def createFeedbackQuery(self, topDoc):
        """Feedback relevant documents using POS tagging to extract nouns and verbs"""
        selectedDoc = None
        for idx, doc in enumerate(self.fileNames):
            if topDoc[0] == doc:
                selectedDoc = self.processedDocs[idx]
                break

        if selectedDoc is None:
            return self.queryVector

        # Use NLTK for POS tagging
        tokens = nltk.word_tokenize(' '.join(selectedDoc.words))
        posTags = nltk.pos_tag(tokens)

        # Extract nouns and verbs
        nounVerbList = [word for word, pos in posTags if pos.startswith('NN') or pos.startswith('VB')]

        feedbackDoc = Document(' '.join(nounVerbList), nounVerbList)
        queryTfVector = self.docVector.buildTfVector(feedbackDoc, self.vectorKeywordIndex, self.weightOpt)
        doc2queryVector = self.docVector.buildIfidfVector(queryTfVector, self.idfVector, self.weightOpt)
        feedbackQuery = self.queryVector + 0.5 * doc2queryVector
        return feedbackQuery

    def feedbackRel(self, feedbackQuery):
        """Re-rank documents using feedback query"""
        scores = []
        for i, docVector in enumerate(self.documentVectors):
            score = util.cosine(feedbackQuery, docVector)
            scores.append((self.fileNames[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:10]

class RankEvaluation(VectorSpace):

    # for every query, store the MAP, MRR, and Recall
    rankList = []
    isRelQueries = []
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
        return 1 if docID.replace("d", "") in self.relDict[self.queryNames[i]] else 0

    def __calculateAP(self, isRelList):
        """ calculate the average precision """
        precisionQuery = 0
        count = 0
        for i, isRel in enumerate(isRelList):
            if isRel:
                count += 1
                precisionQuery += count / (i + 1)
        apQuery = precisionQuery / count if count != 0 else 0
        return apQuery

    def __calculateRR(self, isRelList):
        """ calculate the reciprocal rank """
        for i, isRel in enumerate(isRelList):
            if isRel:
                return 1 / (i + 1)
        return 0

    def __calculateRecall(self, idx, isRelList):
        """ calculate the recall """
        numRel = util.total(isRelList)
        return numRel / self.queryRelDict[self.queryNames[idx]] if self.queryRelDict[self.queryNames[idx]] != 0 else 0

    def __calculateMetric(self, idx, isRelList):
        apScore = self.__calculateAP(isRelList)
        rrScore = self.__calculateRR(isRelList)
        recallScore = self.__calculateRecall(idx, isRelList)
        return apScore, rrScore, recallScore

    def makeRelList(self, queries):
        for i, query in enumerate(queries):
            scores = super().rankDocuments(query)
            self.rankList.append(scores)
            isRel = [self.__checkRel(i, doc) for doc, score in scores]
            self.isRelQueries.append(isRel)

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
