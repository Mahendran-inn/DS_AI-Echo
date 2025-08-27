import pandas as pd
data= pd.read_csv("chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")
df_ml=data.copy()
df_nlp=data.copy() #this is for NLP training

#change date format
df_ml["date"] = pd.to_datetime(df_ml["date"], errors='coerce')

#mapping for ratings
rating_to_sent = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}

df_ml['label'] = df_ml['rating'].map(rating_to_sent)

#splitting categorical and numerical column
num_cols = df_ml.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df_ml.select_dtypes(include=['object', 'category']).columns


from sklearn.preprocessing import LabelEncoder

# Label encode categorical columns (exclude 'review' for ML)
cat_cols_ml = [c for c in cat_cols if c not in ['review','date']]
le = LabelEncoder()
for col in cat_cols_ml:
    df_ml[col] = le.fit_transform(df_ml[col])

################# EDA PART ######################
# What is the distribution of review ratings? 
# Visualization: Bar chart (1 to 5 stars) 
#  Insight: Understand overall sentiment — are users mostly happy or frustrated? 

import matplotlib.pylab as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x='rating', data=df_ml, color='skyblue')  # single color
plt.title('Distribution of Ratings (1 to 5 stars)')
plt.show()


# How many reviews were marked as helpful (above a certain 
# threshold)? 
# Visualization: Thumbs up/down count or pie chart 
#  Insight: See how much value users find in reviews, e.g., reviews with more than 10 
# helpful votes. 

threshold = 10
helpful_count = (df_ml['helpful_votes'] > threshold).sum()
not_helpful_count = (df_ml['helpful_votes'] <= threshold).sum()

plt.figure(figsize=(5,4))
plt.pie([helpful_count, not_helpful_count], labels=['Helpful', 'Not Helpful'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title(f'Reviews marked as helpful (>{threshold} votes)')
plt.show()
 

# How has the average rating changed over time? 
# Visualization: Line chart with date on x-axis, average rating on y-axis 
#  Insight: Track user satisfaction over weeks/months. 

df_ml['date'] = pd.to_datetime(data['date'])  # reload date
avg_rating_time = df_ml.groupby(df_ml['date'].dt.to_period('M'))['rating'].mean()

avg_rating_time.plot(kind='line', figsize=(10,5), marker='o')
plt.title('Average Rating Over Time')
plt.ylabel('Average Rating')
plt.xlabel('Month')
plt.show()


# How do ratings vary by user location? 
# Visualization: Bar chart or world map 
#  Insight: Identify regional differences in satisfaction or experience. 

loc_avg = df_ml.groupby('location')['rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,5))
sns.barplot(x=loc_avg.index, y=loc_avg.values)
plt.xticks(rotation=45)
plt.title('Average Ratings by Location')
plt.show()


# Which platform (Web vs Mobile) gets better reviews? 
# Visualization: Grouped bar chart comparing average ratings by platform 
#  Insight: Helps product teams focus improvements. 

plt.figure(figsize=(6,4))
sns.barplot(x='platform', y='rating', data=df_ml, errorbar=None)
plt.title('Average Ratings by Platform')
plt.show()


# Are verified users more satisfied than non-verified ones? 
# Visualization: Pie chart or side-by-side bar chart comparing rating averages 
#  Insight: Indicates whether loyal/paying users are happier. 

plt.figure(figsize=(6,4))
sns.barplot(x='verified_purchase', y='rating', data=df_ml, errorbar=None)
plt.title('Verified vs Non-verified User Ratings')
plt.show()


# What’s the average length of reviews per rating category? 
# Visualization: Box plot or bar chart 
#  Insight: Shows whether people write longer reviews when they're unhappy or very 
# happy. 

# Ensure 'review' column is string
df_ml['review'] = df_ml['review'].astype(str)

# Create review length safely
df_ml['review_length'] = df_ml['review'].str.len()

# Plot review length per rating
plt.figure(figsize=(6,4))
sns.boxplot(x='rating', y='review_length', data=df_ml)
plt.title('Review Length per Rating')
plt.show()


# What ChatGPT version received the highest average rating? 
# Visualization: Bar chart (version vs. average rating) 
#  Insight: Evaluate improvement or regression across updates.

# Aggregate average rating per version
version_avg = df_ml.groupby('version')['rating'].mean().sort_values(ascending=False)

# Option 1: Show only top 15 versions by rating
top_version_avg = version_avg.head(15)

plt.figure(figsize=(12,5))
sns.barplot(x=top_version_avg.index, y=top_version_avg.values)
plt.xticks(rotation=45, ha='right')  # Rotate and align labels
plt.title('Average Rating per ChatGPT Version (Top 15)')
plt.tight_layout()  # Prevent label cutoff
plt.show()

############################### Machine Learning ##################################################################

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x = df_ml.drop(['label','review','date'], axis=1)
y = df_ml['label']

from sklearn.model_selection import train_test_split, GridSearchCV
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


lr=LogisticRegression()
knn=KNeighborsClassifier()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Logistic Regression pipeline
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

lr_para = {
    'lr__fit_intercept': [True, False],
    'lr__max_iter': [200, 500, 1000],
    'lr__solver': ['lbfgs', 'saga']
}

# KNN pipeline
pipe_knn = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('knn', KNeighborsClassifier())
])

knn_para = {
    'knn__n_neighbors': [3, 5, 7, 10],
    'knn__weights': ['uniform', 'distance']
}


# Logistic Regression
grid_lr = GridSearchCV(pipe_lr, lr_para, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(xtrain, ytrain)

# KNN
grid_knn = GridSearchCV(pipe_knn, knn_para, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(xtrain, ytrain)


# # Best params and scores
# print("Best Logistic Regression:", grid_lr.best_params_, "Score:", grid_lr.best_score_)
# print("Best KNN:", grid_knn.best_params_, "Score:", grid_knn.best_score_)

############################ Deep Learning ###############################################################################
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# 1. Features & Labels
X_dl = df_ml.drop(['label','review','date'], axis=1).values   # features
y_dl = df_ml['label'].values  # labels (Negative/Neutral/Positive)

# Encode labels into integers
le_dl = LabelEncoder()
y_encoded = le_dl.fit_transform(y_dl)   # e.g: Negative=0, Neutral=1, Positive=2

# Convert to one-hot (needed for softmax output layer)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_categorical, test_size=0.2, random_state=42)

# Scale features
scaler_dl = StandardScaler()
X_train_dl = scaler_dl.fit_transform(X_train_dl)
X_test_dl = scaler_dl.transform(X_test_dl)


# 2. Deep Learning Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_dl.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_dl.shape[1], activation='softmax')
])


# 3. Compile & Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    X_train_dl, y_train_dl, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=1)


# 4. Evaluate
loss, acc = model.evaluate(X_test_dl, y_test_dl, verbose=0)
# print(f"DL Test Accuracy: {acc:.4f}")



# History plots (DL training monitoring)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


################################# Sentiment Analysis ##########################################################################

# sentiment_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Sentiment Scoring
df_nlp['scores'] = df_nlp['review'].astype(str).apply(lambda x: sia.polarity_scores(x))
df_nlp['compound'] = df_nlp['scores'].apply(lambda x: x['compound'])

def get_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df_nlp['predicted_sentiment'] = df_nlp['compound'].apply(get_sentiment)


# Streamlit UI
st.title("ChatGPT Review Sentiment Analysis")

menu = [
    "1. Overall Sentiment Distribution",
    "2. Sentiment vs Rating",
    "3. Keywords by Sentiment",
    "4. Sentiment Over Time",
    "5. Verified vs Non-Verified Sentiment",
    "6. Review Length vs Sentiment",
    "7. Sentiment by Location",
    "8. Sentiment by Platform",
    "9. Sentiment by ChatGPT Version",
    "10. Negative Feedback Themes"
]

choice = st.sidebar.selectbox("Select Analysis Question", menu)

# Q1: Overall Sentiment Distribution
if choice == "1. Overall Sentiment Distribution":
    st.subheader("Overall Sentiment Distribution")
    counts = df_nlp['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, palette='pastel', ax=ax)
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)


# Q2: Sentiment vs Rating
elif choice == "2. Sentiment vs Rating":
    st.subheader("Sentiment vs Rating")
    if 'rating' in df_nlp.columns:
        crosstab = pd.crosstab(df_nlp['rating'], df_nlp['predicted_sentiment'], normalize='index')*100
        st.dataframe(crosstab.round(2))
        st.bar_chart(crosstab)


# Q3: Keywords by Sentiment
elif choice == "3. Keywords by Sentiment":
    st.subheader("Keywords by Sentiment (WordClouds)")
    for sentiment in df_nlp['predicted_sentiment'].unique():
        text = " ".join(df_nlp[df_nlp['predicted_sentiment']==sentiment]['review'].astype(str))
        if text.strip():
            wc = WordCloud(width=600, height=400, background_color="white").generate(text)
            st.write(f"### {sentiment} Reviews WordCloud")
            st.image(wc.to_array())


# Q4: Sentiment Over Time
elif choice == "4. Sentiment Over Time":
    st.subheader("Sentiment Over Time")
    if 'date' in df_nlp.columns:
        df_nlp['date'] = pd.to_datetime(df_nlp['date'], errors='coerce')
        timeline = df_nlp.groupby([pd.Grouper(key='date', freq='M'), 'predicted_sentiment']).size().unstack()
        st.line_chart(timeline.fillna(0))


# Q5: Verified vs Non-Verified
elif choice == "5. Verified vs Non-Verified Sentiment":
    st.subheader("Verified vs Non-Verified Sentiment")
    if 'verified_purchase' in df_nlp.columns:
        verified_dist = pd.crosstab(df_nlp['verified_purchase'], df_nlp['predicted_sentiment'], normalize='index')*100
        st.bar_chart(verified_dist)


# Q6: Review Length vs Sentiment
elif choice == "6. Review Length vs Sentiment":
    st.subheader("Review Length vs Sentiment")
    df_nlp['review_length'] = df_nlp['review'].astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots()
    sns.boxplot(x='predicted_sentiment', y='review_length', data=df_nlp, ax=ax, palette="Set2")
    ax.set_ylabel("Number of Words in Review")
    st.pyplot(fig)


# Q7: Sentiment by Location
elif choice == "7. Sentiment by Location":
    st.subheader("Sentiment by Location")
    if 'location' in df_nlp.columns:
        loc_dist = pd.crosstab(df_nlp['location'], df_nlp['predicted_sentiment'], normalize='index')*100
        st.bar_chart(loc_dist)


# Q8: Sentiment by Platform
elif choice == "8. Sentiment by Platform":
    st.subheader("Sentiment by Platform")
    if 'platform' in df_nlp.columns:
        plat_dist = pd.crosstab(df_nlp['platform'], df_nlp['predicted_sentiment'], normalize='index')*100
        st.bar_chart(plat_dist)


# Q9: Sentiment by ChatGPT Version
elif choice == "9. Sentiment by ChatGPT Version":
    st.subheader("Sentiment by ChatGPT Version")
    if 'version' in df_nlp.columns:
        version_dist = pd.crosstab(df_nlp['version'], df_nlp['predicted_sentiment'], normalize='index')*100
        st.bar_chart(version_dist)


# Q10: Negative Feedback Themes
elif choice == "10. Negative Feedback Themes":
    st.subheader("Negative Feedback Themes (WordCloud)")
    negative_text = " ".join(df_nlp[df_nlp['predicted_sentiment']=="Negative"]['review'].astype(str))
    if negative_text.strip():
        wc = WordCloud(width=800, height=500, background_color="white", colormap="Reds").generate(negative_text)
        st.image(wc.to_array())
