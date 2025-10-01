from pprint import pprint
from Parser import Parser
import util
import os
import numpy as np
import math

def read_news_from_folder(folder_path):
    # 獲取目錄下所有的.txt檔案
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x[4:-4]))
    news_list = []

    for txt_file in txt_files:
        with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as file:
            content = file.read()
            news_list.append(content)
    return news_list

def df(A):
    A = np.array(A)
    return np.sum(A > 0, axis=0).tolist()

def idf(A,vectorSpace):
    total_num_of_doc = len(vectorSpace.documentVectors[0])
    for i in range(total_num_of_doc):
        A[i] = math.log((total_num_of_doc/A[i]),10)
    return A

def tf_idf(A, B):
    result = []
    for row in A:
        multiplied_row = [row[i] * B[i] for i in range(len(B))]
        result.append(multiplied_row)
    return result

def top_scores(A, B, n=10):
    # 使用zip組合A和B，然後根據分數進行排序
    sorted_pairs = sorted(zip(A, B), key=lambda x: x[1], reverse=True)

    # 取出前n個元素
    top_n = sorted_pairs[:n]

    # 格式化輸出
    print("top", n, "        score")
    for pair in top_n:
        print(pair[0], " ", pair[1])
    
    return top_n

def bottom_scores(A, B, n=10):
    # 使用zip組合A和B，然後根據分數進行排序
    sorted_pairs = sorted(zip(A, B), key=lambda x: x[1], reverse=False)

    # 取出前n個元素
    bottom_n = sorted_pairs[:n]
    # bottom_n = [('News561.txt', 3.3515547509817782), ('News13136.txt', 6.191829687027521),...]
    # 格式化輸出
    print("bottom", n, "        score")
    for pair in bottom_n:
        print(pair[0], "       ", pair[1])
def search_bycosine(array):
    queryVector = array
    ratings = [util.cosine(queryVector, documentVector) for documentVector in vectorSpace.documentVectors]
    #ratings.sort(reverse=True)
    return ratings

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]

        #print(self.vectorKeywordIndex)
        #print(self.documentVectors)


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
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings


    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList) #會把搜尋字根據keyIndex來變成向量
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings
    
    def Euclidean_Distance(self,searchList):
      B = self.buildQueryVector(searchList) #會把搜尋字根據keyIndex來變成向量
      A = vectorSpace.documentVectors
      ans = [sum((a-b)**2 for a, b in zip(vec, B))**0.5 for vec in A]
      return ans
    
    def getQueryVector(self,searchList):
        return self.buildQueryVector(searchList)


if __name__ == '__main__':
    #test data
    # documents = ["The cat in the hat disabled ",
    #              "A cat is a fine pet ponies.",
    #              "Dogs and cats make good pets.",
    #              "I haven't got a hat."]

    # vectorSpace = VectorSpace(documents)

    #print(vectorSpace.vectorKeywordIndex)

    # print(vectorSpace.documentVectors)

    # for i in range(4):
    #   print(vectorSpace.documentVectors[i])
    #   print(vectorSpace.related(i),"\n\n")

    # print(vectorSpace.search(["The cat in the hat disabled "]))

    #-----------------------------------------------------------------以上做測試並理解程式碼
    #--------------------- TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity---------------------
    #讀取News為all_news
    folder_path = "EnglishNews"
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x[4:-4]))
    all_news = read_news_from_folder(folder_path)

    #建立向量空間
    vectorSpace = VectorSpace(all_news)

    #7875份檔案 共19457關鍵字 所以每篇news共有19457維 也就是每一維即TF(會累積出現次數)

    #計算DF
    df_arr = df(vectorSpace.documentVectors)
    # print(len(df_arr))
    # print(df_arr)
    
    #計算IDF
    idf_arr = idf(df_arr,vectorSpace)
    # print(len(idf_arr))
    # print(idf_arr)
    
    #計算TF-IDF
    tfidf = tf_idf(vectorSpace.documentVectors,idf_arr)
    vectorSpace.documentVectors = tfidf

    #使用cos計算與query的距離
    cos_distance = vectorSpace.search(["Trump Taiwan travel"])

    #輸出結果
    print("這是TF-IDF Weighting + Cosine Similarity的輸出結果")
    top_n = top_scores(txt_files,cos_distance,n=10)
    # print(top_n[0],"\n\n") ('News10640.txt', 0.30064468700651087)
    print("\n\n\n")
    
    ################## TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance
    print("這是TF-IDF Weighting + Euclidean Distance的輸出結果")
    euclidean = vectorSpace.Euclidean_Distance(["Trump Taiwan travel"])
    bottom_scores(txt_files,euclidean,n=10)
    print("\n\n\n")
  #---------------------Relevance Feedback-----------------
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    # 假設 original_query_vector 是原始查詢的向量表示，document_content 是第一篇文件的內容
    original_query_vector = vectorSpace.getQueryVector(["Trump Taiwan travel"])
    first_doc_name = top_n[0][0]
    file_path = 'EnglishNews/' + first_doc_name
    with open(file_path, 'r', encoding='utf-8') as file:
      document_content = file.read()


    # 使用 NLTK 進行詞性標註
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    words = word_tokenize(document_content)
    tagged_words = nltk.pos_tag(words)

    # 從第一篇文章中提取名詞和動詞
    relevant_words = [word for word, pos in tagged_words if pos in ["NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
    # print(relevant_words) ['travel', 'alert',...]

    # 建立二次查詢向量
    query_vector2 = vectorSpace.getQueryVector(relevant_words)

    # 計算新的查詢向量
    new_query_vector = [original_query_vector[i] + 0.5 * query_vector2[i] for i in range(len(query_vector2))]
    

    #使用新的向量與再去搜索一次文章，使用cos計算文章與query的距離
    cos_distance = search_bycosine(new_query_vector)

    #輸出結果
    print("這是Relevance Feedback + Cosine Similarity的輸出結果")
    top_n = top_scores(txt_files,cos_distance,n=10)



    
###################################################
#筆記區：
# 先把多個文章陣列化，放入vectorSpace = VectorSpace(documents)

# 執行print(vectorSpace.vectorKeywordIndex) 將重點文字以json的方式列出 eg.{'pet': 0, 'make': 1, 'dog': 2, 'cat': 3, 'poni': 4, 'good': 5, 'hat': 6, 'fine': 7, 'disabl': 8}

# 若再執行print(vectorSpace.documentVectors) 可以依據上方json把文章向量化 印出9維的向量 並計算出現次數
# eg.[[0, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]

#若執行print(vectorSpace.related(1)) 1號文章 與 0~3號文章計算cos數值 並顯示為[0.25, 1.0, 0.4472135954999579, 0.0] ps.一號位置為1很合理 因為是同篇文章所以 cos=1

#vectorSpace.search("cat")這個可以將cat根據keyindex變成向量，並且和每個docVec計算cos後存入陣列中，所以說n個文章陣列就有0~n-1

#基本運作方法：
#1. vectorSpace = VectorSpace(documents) 編碼文章
#2. vectorSpace.vectorKeywordIndex 根據文章產生keyword
#3. vectorSpace.related 可以知道文章之間彼此的相關度
#4. vectorSpace.search("cat") 可以對文章進行搜尋
