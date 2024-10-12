import pandas as pd
import ast
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
survey_mapping = pd.read_csv('results/survey_question_mapping.csv')
global_opinions = pd.read_csv('data/data_global_opinions.csv')

# Clean NaN values
survey_mapping = survey_mapping.dropna(subset=['English Question Text'])
global_opinions = global_opinions.dropna(subset=['question'])

# Prepare lists of all questions
survey_questions = survey_mapping['English Question Text'].tolist()
global_questions = global_opinions['question'].tolist()

# Vectorize all survey and global questions
vectorizer = TfidfVectorizer().fit(survey_questions + global_questions)
survey_vectors = vectorizer.transform(survey_questions)
global_vectors = vectorizer.transform(global_questions)

# Compute pairwise similarity
similarity_matrix = cosine_similarity(survey_vectors, global_vectors)

# Initialize a list to hold the results
results = []

# Threshold for similarity
similarity_threshold = 0.95

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # If parsing fails, try to fix common issues
        val = val.replace("\n", " ")  # Replace newline characters
        val = val.strip()  # Remove leading/trailing whitespace
        if not (val.startswith("[") and val.endswith("]")):
            # If it still doesn't look like a list, return an empty list or handle appropriately
            return []
        try:
            return ast.literal_eval(val)
        except:
            # If it still fails, return an empty list or log the error
            return []
        
# Iterate over survey questions and their similarities to all global questions
for i, survey_row in survey_mapping.iterrows():
    survey_question_id = survey_row['English Question ID']
    survey_question_text = survey_row['English Question Text']
    survey_options = safe_literal_eval(survey_row['English Response Options'])  # Convert string representation to list

    if i >= len(similarity_matrix):
        print(f"Index {i} out of bounds for similarity matrix. Skipping...")
        continue
    # Get similarity scores for this survey question with all global questions
    similarity_scores = similarity_matrix[i]

    # Iterate over global questions and their corresponding similarity scores
    for j, similarity_score in enumerate(similarity_scores):
        if similarity_score >= similarity_threshold:
            global_row = global_opinions.iloc[j]
            global_question_text = global_row['question']

            # Extract selections and options from the global data
            try:
                selections = re.sub(r"defaultdict\(.*?, ", "", global_row['selections']).rstrip(")")
                selections = safe_literal_eval(selections)
                options = safe_literal_eval(global_row['options'])
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing selections or options: {e}")
                continue

            # Check if 'Turkey' is in selections
            if 'Turkey' in selections:
                turkey_data = selections['Turkey']

                # Find the index of the highest selection value for Turkey
                max_index = turkey_data.index(max(turkey_data))

                # Get the corresponding response option
                response = options[max_index]

                # Append the relevant data to the results list
                results.append({
                    'Survey Question ID': survey_question_id,
                    'Survey Question English Text': survey_question_text,
                    'Global Data Question Text': global_question_text,
                    'Similarity Score': similarity_score,
                    'Survey Options': survey_options,
                    'Global Data Options': options,
                    'Turkey Selection Data': turkey_data,
                    'Selected Response': response
                })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save to a new CSV file
results_df.to_csv('results/matched_survey_global_data.csv', index=False)

print("Data matching complete and saved to 'results/matched_survey_global_data.csv'.")

