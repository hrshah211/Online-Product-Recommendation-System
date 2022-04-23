from flask import Flask, render_template, request
import pymongo
import pandas as pd

from pandas import DataFrame
from sklearn import neighbors 
from sklearn.neighbors import NearestNeighbors
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import string

username = "artificial_brilliance"
password = "pass123"
database = "Amazon_Customer_Product_Dataset"
db_url = "mongodb+srv://"+username+":"+password+"@cluster0.rooti.mongodb.net/"+database+"?retryWrites=true&w=majority"

client = pymongo.MongoClient(db_url)
db = client[database]

col = db["Customer_Review"]
x = col.find({})
lis=[]
 
for data in x:
    del data['_id']
    lis.append(data)

reviews = pd.DataFrame(lis)

col = db["Product_Info"]
x = col.find({})
lis=[]

for data in x:
    del data['_id']
    lis.append(data)

products = pd.DataFrame(lis)
categories = products['category'].unique()
ids = []
for i in range(0,len(categories)):
    ids.append(i+1)

dfReviews_train = ''
df3 = ''
regEx = re.compile('[^a-z]+')
def cleanReviews(reviewText):
    reviewText = reviewText.lower()
    reviewText = regEx.sub(' ', reviewText).strip()
    return reviewText

def setup(category, products, reviews):
    categorisedproducts = products[products['category'] == category]
    df = pd.merge(categorisedproducts, reviews, on=["asin"])
    count = df.groupby("asin", as_index=False).count()

    dfMerged = pd.merge(df, count, how='right', on=['asin'])

    dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
    dfMerged["overallScore"] = dfMerged["overall_x"]
    dfMerged["summaryReview"] = dfMerged["summary_x"]
    dfMerged["category"] = dfMerged["category_x"]


    dfProductReview = df.groupby("asin", as_index=False).mean()
    ProductReviewSummary = dfMerged.groupby("asin")["summaryReview"].apply(list)
    ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
    ProductReviewSummary.to_csv("ProductReviewSummary.csv")

    categoriseddf = pd.merge(dfProductReview,products, on='asin', how='left')
    categoriseddf = categoriseddf[['asin','overall','category']]
    global df3
    df3 = pd.read_csv("ProductReviewSummary.csv")
    df3 = pd.merge(df3, categoriseddf, on="asin", how='inner')

    df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)
    df3 = df3.reset_index()

    reviews = df3["summaryClean"] 
    countVector = CountVectorizer(max_features = 300, stop_words='english') 
    transformedReviews = countVector.fit_transform(reviews) 

    dfReviews = pd.DataFrame(transformedReviews.A, columns=countVector.get_feature_names())
    dfReviews = dfReviews.astype(int)
    dfReviews['category'] = df3['category']

    encoding = {'Arts Crafts And Sewing':1,'Appliances':2, 'Fashion':3, 'Beauty':4}
    for i in range(0,len(dfReviews)):
      dfReviews['category'][i] = encoding[dfReviews['category'][i]]

    X = np.array(dfReviews)
    global dfReviews_train
    dfReviews_train = X

    neighbor = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(dfReviews_train)
    return neighbor

def getRecommendations(product, model):
    print(df3.index[df3['asin'] == product])
    index = df3.index[df3['asin'] == product]
    res = model.kneighbors([dfReviews_train[index][0]])[1]
    recommendations = []
    for i in range(1,6):
        recommendations.append(df3['asin'][res[0][i]])
    recommended_products = pd.DataFrame()
    for id in recommendations:
        prod = pd.DataFrame(products[products['asin'] == id][['title','price', 'imageURLHighRes']])
        recommended_products = pd.concat([prod, recommended_products])
    recommendations_title = recommended_products['title'].tolist()
    recommendations_price = recommended_products['price'].tolist()
    recommendations_image = recommended_products['imageURLHighRes'].tolist()
    return recommendations_title, recommendations_price, recommendations_image

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html",categories=categories, ids = ids)

@app.route("/loadproducts", methods=["GET", "POST"])
def loadProducts():
    index = int(request.data) - 1
    category = categories[index]
    selected_products = products[products['category'] == category]
    selected_products = selected_products.sample(n = 3)
    imageURL = selected_products['imageURLHighRes'].tolist()
    titles = selected_products['title'].tolist()
    price = selected_products['price'].tolist()
    asin = selected_products['asin'].tolist()
    neighbor_model = setup(category, products, reviews)
    recone_title, recone_price, recone_image = getRecommendations(asin[0], neighbor_model)
    rectwo_title, rectwo_price, rectwo_image = getRecommendations(asin[1], neighbor_model)
    recthree_title, recthree_price, recthree_image =  getRecommendations(asin[2], neighbor_model)
    return render_template("products.html",category=category, titles = titles, price = price, imageURL = imageURL,
    recone_title = recone_title, recone_price = recone_price, recone_image = recone_image,
    rectwo_title = rectwo_title, rectwo_price = rectwo_price, rectwo_image = rectwo_image,
    recthree_title = recthree_title, recthree_price = recthree_price, recthree_image = recthree_image)