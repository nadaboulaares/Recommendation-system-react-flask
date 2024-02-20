from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS module
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib

app = Flask(__name__)
CORS(app, resources={r"/recommend": {"origins": "http://localhost:3000"}})  # Adjust the origin to match your React app's URL

# Load the saved components of the model
count_vectorizer = joblib.load('count_vectorizer.joblib')
svd = joblib.load('svd.joblib')
emotions_with_user_info = joblib.load('emotions_with_user_info.joblib')

# Select relevant columns for content-based recommendation
selected_columns = ['Emotion', 'Age', 'Gender', 'medical_state', 'Energy_level', 'Duration']
selected_df = emotions_with_user_info[selected_columns].copy()

# Convert categorical columns to strings for text-based processing
selected_df['Gender'] = selected_df['Gender'].astype(str)
selected_df['medical_state'] = selected_df['medical_state'].astype(str)

# Create a feature matrix by combining relevant columns
selected_df['features'] = selected_df.apply(lambda row: ' '.join(row[['Emotion', 'Age', 'Gender', 'medical_state']].astype(str)), axis=1)

# Initialize the CountVectorizer to convert text data into a matrix of token counts
count_vectorizer = CountVectorizer()
feature_matrix = count_vectorizer.fit_transform(selected_df['features'])

# Perform matrix factorization to reduce the dimensionality of the feature matrix
svd = TruncatedSVD(n_components=10)
latent_matrix = svd.fit_transform(feature_matrix)

# Function to get recommendations based on input features
def get_recommendations(input_emotion, input_age, input_gender, input_medical_state, num_recommendations=5):
    input_features = [input_emotion, input_age, input_gender, input_medical_state]
    input_features = ' '.join(map(str, input_features))
    input_df = pd.DataFrame([input_features], columns=['features'])  # Create a DataFrame to match the selected_df structure

    input_feature_matrix = count_vectorizer.transform(input_df['features'])
    input_latent_matrix = svd.transform(input_feature_matrix)

    sim_scores = cosine_similarity(input_latent_matrix, latent_matrix)
    sim_scores = list(sim_scores[0])
    sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    recommended_indices = [index for index, _ in sim_scores]
    return emotions_with_user_info.iloc[recommended_indices][['Energy_level', 'Duration']]

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    input_emotion = data['emotion']
    input_age = data['age']
    input_gender = data['gender']
    input_medical_state = data['medical_state']
    num_recommendations = data.get('num_recommendations', 5)

    recommended_videos = get_recommendations(input_emotion, input_age, input_gender, input_medical_state, num_recommendations)

    # Convert the result to JSON and return it
    result = recommended_videos.to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
