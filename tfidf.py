from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import json
import pandas as pd
import csv
from nltk.stem import PorterStemmer

ps = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (ps.stem(w) for w in analyzer(doc))


# docIDs = {}
count = 0
siteCounter = 0
for site in os.listdir("/Users/nisha/Downloads/DEV"):
    print(site)
    siteCounter += 1
    # if(siteCounter == 7):
    #     break
    print(siteCounter)
    if(site != ".DS_Store"):
        for jsonFile in os.listdir("/Users/nisha/Downloads/DEV/" + site):
            count += 1
            with open("/Users/nisha/Downloads/DEV/" + site + "/" + jsonFile, 'r') as f:
                obj = json.load(f)
                # docIDs[obj["url"]] = count
                try:
                    cv = CountVectorizer(analyzer=stemmed_words)
                    wordcount = cv.fit_transform([obj["content"]])
                    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
                    tfidf_transformer.fit(wordcount)

                    countvect = cv.transform([obj["content"]])
                    tfidfvect = tfidf_transformer.transform(countvect)

                    feature_names = cv.get_feature_names()
                    df = pd.DataFrame(tfidfvect[0].T.todense(), index=feature_names)
                    f = open("TF-IDF/"+ str(count // 1000) + ".csv", "a+")
                    writer = csv.writer(f)
                    writer.writerow([count, df.to_dict("index")])
                except ValueError:
                    print("val error", count)
