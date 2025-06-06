# '''
# This is the main function of the URL malware detection of this program
# '''

# import pandas as pd
# import numpy as np
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# import dill as pickle


# def sanitization(web: str) -> list:
#     web = web.lower()
#     token = []
#     dot_token_slash = []
#     raw_slash = str(web).split('/')
#     for i in raw_slash:
#         # removing slash to get token
#         raw1 = str(i).split('-')
#         slash_token = []
#         for j in range(0, len(raw1)):
#             # removing dot to get the tokens
#             raw2 = str(raw1[j]).split('.')
#             slash_token = slash_token + raw2
#         dot_token_slash = dot_token_slash + raw1 + slash_token
#     # to remove same words and 'com'
#     token = list(set(dot_token_slash))
#     if 'com' in token:
#         # remove com
#         token.remove('com')
#     return token


# # Get URL input
# url = str(input("Input the URL that you want to check (eg. google.com): "))

# # Using whitelist filter as the model fails in some legit cases
# file = 'D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_URL_whitelist.pkl'
# with open(file, 'rb') as f:
#     whitelist = pickle.load(f)
# f.close()
# # print(whitelist)
# s_url = [url if url not in whitelist else '']


# # Loading the Linear Regression model
# file = "D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_model.pkl"
# with open(file, 'rb') as f1:
#     lgr = pickle.load(f1)
# f1.close()

# # Loading the vectorizer
# file = "D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_vector.pkl"
# with open(file, 'rb') as f2:
#     vectorizer = pickle.load(f2)
# f2.close()

# # Transform url to Tf-idf-weighted document-term matrix
# x = vectorizer.transform(s_url)
# # Predicting url
# y_predict = lgr.predict(x)

# # Print the result
# print(y_predict[0])


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dill as pickle


def sanitization(web):
    web = web.lower()
    token = []
    dot_token_slash = []
    raw_slash = str(web).split('/')
    for i in raw_slash:
        # removing slash to get token
        raw1 = str(i).split('-')
        slash_token = []
        for j in range(0, len(raw1)):
            # removing dot to get the tokens
            raw2 = str(raw1[j]).split('.')
            slash_token = slash_token + raw2
        dot_token_slash = dot_token_slash + raw1 + slash_token
    # to remove same words and 'com'
    token = list(set(dot_token_slash))
    if 'com' in token:
        # remove com
        token.remove('com')
    return token


# Get URL input
url = str(input("Input the URL that you want to check (eg. google.com): "))

# Using whitelist filter as the model fails in some legit cases
file = 'D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_URL_whitelist.pkl'
with open(file, 'rb') as f:
    whitelist = pickle.load(f)
f.close()
# Check if URL is in whitelist
if url in whitelist:
    print("Safe URL (in whitelist)")
else:
    # Loading the Linear Regression model
    file = "D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_model.pkl"
    with open(file, 'rb') as f1:
        lgr = pickle.load(f1)
    f1.close()

    # Loading the vectorizer
    file = "D:/Viren/AIKD/Malware/malware-dection/Classifier/URL_Detector/pickel_vector.pkl"
    with open(file, 'rb') as f2:
        vectorizer = pickle.load(f2)
    f2.close()

    # Manually apply your sanitization function
    tokenized_url = ' '.join(sanitization(url))
    
    # Transform url to Tf-idf-weighted document-term matrix
    x = vectorizer.transform([tokenized_url])
    
    # Predicting url
    y_predict = lgr.predict(x)

    # Print the result
    print(y_predict[0])