import os
import nltk
import argparse
import VectorSpace

def readDocuments(directory_path):
    fileNames = []
    documents = []
    for fileName in os.listdir(directory_path):
        with open(os.path.join(directory_path, fileName), 'r', encoding='utf-8') as file:
            documents.append(file.read())
            fileNames.append(fileName.replace('.txt',''))
        return documents, fileNames

def main():
    parser = argparse.ArgumentParser(description="Process English and Chinese queries.")

    parser.add_argument('--Eng_query', type=str, default="planet Taiwan typhoon", help='Input English query (default: "planet Taiwan typhoon")')
    parser.add_argument('--Chi_query', type=str, default="音樂 科技", help='Input Chinese query (default: "音樂 科技")')

    args = parser.parse_args()

    engQuery = args.Eng_query
    chiQuery = args.Chi_query

    # nltk.download('punkt') link:  https://github.com/nltk/nltk/issues/3293
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

    # test data
    # documents = ["The cat in the hat disabled??", "A HE it IT he cat is a fine pet ponies.", "Dogs and cats make good pets.", "I haven't got a hat."]
    # fileNames = ["New1", "New2", "New3", "New4"]

    # Task1
    ################################
    print(f"English Query: {engQuery}")
    print(f"Chinese Query: {chiQuery}")
    print("Task 1")
    directoryPath = "./EnglishNews/EnglishNews/"
    documents, fileNames = readDocuments(directoryPath)
    print("Raw-TF Cosine")
    print('%-11s' % "NewID", "Score")
    VSMTask1 = VectorSpace(documents, fileNames, "RawTF", "cosine", "EN")
    scoresTask1 = VSMTask1.rankDocuments(engQuery)
    for NewID, score in scoresTask1:
        print('%-11s' % NewID, score)
    print('---------------------------------')
    print("TF-IDF Cosine")
    print('%-11s' % "NewID", "Score")
    VSMTask1 = VectorSpace(documents, fileNames, "TFIDF", "cosine", "EN")
    VSMTask1 = VSMTask1.rankDocuments(engQuery)
    for NewID, score in VSMTask1:
        print('%-11s' % NewID, score)
    print('---------------------------------')
    print("Raw-TF Euclidean")
    print('%-11s' % "NewID", "Score")
    VSMTask1 = VectorSpace(documents, fileNames, "RawTF", "Euclidean", "EN")
    scoresTask1 = VSMTask1.rankDocuments(engQuery)
    for NewID, score in scoresTask1:
        print('%-11s' % NewID, score)
    print('---------------------------------')
    print("TF-IDF Euclidean")
    print('%-11s' % "NewID", "Score")
    VSMTask1 = VectorSpace(documents, fileNames, "TFIDF", "Euclidean", "EN")
    scoresTask1 = VSMTask1.rankDocuments(engQuery)
    for NewID, score in scoresTask1:
        print('%-11s' % NewID, score)
    VSMTask1 = None
    scoresTask1 = None
    print("############################################")
    ################################

    # Task2
    ################################
    print("Task 2")
    directoryPath = "./EnglishNews/EnglishNews/"
    documents, fileNames = readDocuments(directoryPath)
    print("TF-IDF Cosine")
    print('%-11s' % "NewID", "Score")
    VSMTask2 = VectorSpace(documents, fileNames, "TFIDF", "cosine", "EN")
    scoresTask2 = VSMTask2.rankDocuments(engQuery)
    for NewID, score in scoresTask2:
        print('%-11s' % NewID, score)
    print('---------------------------------')
    print("TF-IDF Cosine plus feedback")
    feedbackQuery = VSMTask2.createFeedackQuery(scoresTask2[0])
    scoresTask2 = VSMTask2.feedbackRel(feedbackQuery)
    for NewID, score in scoresTask2:
        print('%-11s' % NewID, score)
    VSMTask2 = None
    scoresTask2 = None
    print("############################################")
    ################################

    # test data    
    # documents = ["這是一篇關於人工智慧的新聞文章", "這是另一篇關於機器學習的新聞", "這是與政治有關的新聞文章", "假新聞"]
    # fileNames = ["New1", "New2", "New3", "New4"]
    
    # Task3
    ################################
    print("Task 3")
    directoryPath = "./ChineseNews/ChineseNews/"
    documents, fileNames = readDocuments(directoryPath)
    VSMTask3 = VectorSpace(documents, fileNames, "RawTF", "cosine", "CN")
    scoresTask3 = VSMTask3.rankDocuments(chiQuery)
    print("Raw-TF Cosine")
    print('%-11s' % "NewID", "Score")
    for NewID, score in scoresTask3:
        print('%-11s' % NewID, score)
    print('---------------------------------')
    VSMTask3 = VectorSpace(documents, fileNames, "TFIDF", "cosine", "CN")
    scoresTask3 = VSMTask3.rankDocuments(chiQuery)
    print("TF-IDF Cosine")
    print('%-11s' % "NewID", "Score")
    for NewID, score in scoresTask3:
        print('%-11s' % NewID, score)
    VSMTask3 = None
    scoresTask3 = None
    print("############################################")
    ################################

    # Task4
    ################################
    print("Task 4")
    directoryPath = "./smaller_dataset/smaller_dataset/"
    documentPath = "collections/"
    queryPath = "queries/"
    relPath = "rel.tsv"
    documents, fileNames = readDocuments(directoryPath + documentPath)
    queries, queryNames = readDocuments(directoryPath + queryPath)
    rel = readRel(directoryPath + relPath)
    VSMTask4 = RankEvaluation(documents, fileNames, "TFITF", "cosine", "EN", queryNames, rel)
    # print(len(VSMTask4a.documentVectors)) # 1460
    # print(VSMTask4a.vectorKeywordIndex) # len = 8305
    VSMTask4.makeRelList(queries)
    print("TF-IDF Cosine")
    VSMTask4.evaluate()
    VSMTask4 = None
    print("############################################")

if __name__ == "__main__":
    main()