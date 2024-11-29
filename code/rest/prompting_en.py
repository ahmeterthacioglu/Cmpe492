import ast
import re
import pandas as pd
import numpy as np  # Import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the processed persona data from the previous steps
persona_df = pd.read_csv('results/rest/important_persona_counts_en.csv')
survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')
survey_mapping = pd.read_csv('results/survey_question_mapping.csv')

# Define mappings and helper functions
def categorize_age(age):
    if age <= 29:
        return 'Up to 29'
    elif 30 <= age <= 49:
        return '30-49'
    else:
        return '50 and more'
    
# Ensure that necessary columns are created in survey_data
def preprocess_survey_data(survey_data):
    # Define mappings and helper functions
    gender_map = {1: 'Male', 2: 'Female'}
    marital_status_map = {1: 'Married', 6: 'Single', 3: 'Divorced'}
    
    def categorize_age(age):
        if age <= 29:
            return 'Up to 29'
        elif 30 <= age <= 49:
            return '30-49'
        else:
            return '50 and more'

    def categorize_children(children):
        if pd.isna(children):
            return None
        return '4 or more children' if children >= 4 else f'{int(children)} child{"ren" if children > 1 else ""}'

    # Apply mappings
    survey_data['Gender'] = survey_data['Q260'].map(gender_map)
    survey_data['Age Group'] = survey_data['Q262'].apply(categorize_age)
    survey_data['Marital Status'] = survey_data['Q273'].map(marital_status_map)
    survey_data['Children Group'] = survey_data['Q274'].apply(categorize_children)
    survey_data['Education Level'] = survey_data['Q275'].map({
        0: 'Early childhood education (ISCED 0) / no education',
        1: 'Primary education (ISCED 1)',
        2: 'Lower secondary education (ISCED 2)',
        3: 'Upper secondary education (ISCED 3)',
        4: 'Post-secondary non-tertiary education (ISCED 4)',
        5: 'Short-cycle tertiary education (ISCED 5)',
        6: 'Bachelor or equivalent (ISCED 6)',
        7: 'Master or equivalent (ISCED 7)',
        8: 'Doctoral or equivalent (ISCED 8)'
    })
    survey_data['Employment Status'] = survey_data['Q279'].map({
        1: 'Full-time employee (30 hours a week or more)',
        2: 'Part-time employee (less than 30 hours a week)',
        3: 'Self-employed',
        4: 'Retired/pensioned',
        5: 'Housewife not otherwise employed',
        6: 'Student',
        7: 'Unemployed',
        8: 'Other (write in)'
    })
    survey_data['Social Class'] = survey_data['Q287'].map({
        1: 'Upper class',
        2: 'Upper middle class',
        3: 'Lower middle class',
        4: 'Working class',
        5: 'Lower class'
    })

    # Drop rows with missing values in the key columns
    survey_data.dropna(subset=[
        'Gender', 'Age Group', 'Marital Status', 'Children Group',
        'Education Level', 'Employment Status', 'Social Class'
    ], inplace=True)

    return survey_data

# Preprocess the survey data
survey_data = preprocess_survey_data(survey_data)

def parse_input(prompt):
    # Initialize features as None
    age, gender, marital_status, children, education, employment, social_class = [None] * 7

    # Extract age
    age_match = re.search(r'\b(\d{1,2})\s*years?\s*old\b', prompt)
    if age_match:
        age = int(age_match.group(1))
        age_group = categorize_age(age)

    # Extract gender
    if 'female' in prompt.lower():
        gender = 'Female'
    elif 'male' in prompt.lower():
        gender = 'Male'

    # Extract marital status
    if 'married' in prompt.lower():
        marital_status = 'Married'
    elif 'single' in prompt.lower():
        marital_status = 'Single'
    elif 'divorced' in prompt.lower():
        marital_status = 'Divorced'

    # Extract number of children
    children_match = re.search(r'(\d+)\s*children?', prompt)
    if children_match:
        num_children = int(children_match.group(1))
        children = '4 or more children' if num_children >= 4 else f'{num_children} child{"ren" if num_children > 1 else ""}'
    elif 'no children' in prompt.lower():
        children = '0 children'

    # Extract education level
    education_levels = [
        'Early childhood education (ISCED 0) / no education', 'Primary education (ISCED 1)', 
        'Lower secondary education (ISCED 2)', 'Upper secondary education (ISCED 3)', 
        'Post-secondary non-tertiary education (ISCED 4)', 'Short-cycle tertiary education (ISCED 5)', 
        'Bachelor or equivalent (ISCED 6)', 'Master or equivalent (ISCED 7)', 
        'Doctoral or equivalent (ISCED 8)'
    ]
    for level in education_levels:
        if level.lower() in prompt.lower():
            education = level
            break

    # Extract employment status
    employment_options = [
        'Full-time employee (30 hours a week or more)', 'Part-time employee (less than 30 hours a week)', 
        'Self-employed', 'Retired/pensioned', 'Housewife not otherwise employed', 'Student', 
        'Unemployed', 'Other (write in)'
    ]
    for option in employment_options:
        if option.lower() in prompt.lower():
            employment = option
            break

    # Extract social class
    social_classes = ['Upper class', 'Upper middle class', 'Lower middle class', 'Working class', 'Lower class']
    for s_class in social_classes:
        if s_class.lower() in prompt.lower():
            social_class = s_class
            break

    # Return extracted features
    return {
        'Gender': gender, 'Age Group': age_group, 'Marital Status': marital_status, 
        'Children Group': children, 'Education Level': education, 
        'Employment Status': employment, 'Social Class': social_class
    }

def match_personas(features):
    # Filter personas that match the provided features, allowing for missing features
    matching_personas = persona_df.copy()
    for feature, value in features.items():
        if value is not None:
            matching_personas = matching_personas[matching_personas[feature] == value]

    return matching_personas


def calculate_response_distribution(question_id, matching_personas):
    # Filter the survey data for matching personas
    filtered_data = survey_data[
        (survey_data['Gender'].isin(matching_personas['Gender'])) &
        (survey_data['Age Group'].isin(matching_personas['Age Group'])) &
        (survey_data['Marital Status'].isin(matching_personas['Marital Status'])) &
        (survey_data['Children Group'].isin(matching_personas['Children Group'])) &
        (survey_data['Education Level'].isin(matching_personas['Education Level'])) &
        (survey_data['Employment Status'].isin(matching_personas['Employment Status'])) &
        (survey_data['Social Class'].isin(matching_personas['Social Class']))
    ]

    # Check if there are any matching records
    if not filtered_data.empty and question_id in filtered_data.columns:
        response_distribution = filtered_data[question_id].value_counts(normalize=True) * 100
        response_distribution = response_distribution.sort_index()
        matching_count = len(filtered_data)
        return response_distribution, matching_count
    else:
        # Return an empty series and zero if there are no matches
        return pd.Series(dtype=float), 0
    
    
# Prepare the list of survey questions
survey_questions = [q for q in survey_mapping['English Question Text'].tolist() if pd.notna(q)]

# Vectorize the survey questions
vectorizer = TfidfVectorizer().fit(survey_questions)
survey_vectors = vectorizer.transform(survey_questions)

def find_question_id_by_text(question_text, similarity_threshold=0.75):
    """
    Finds the most similar question ID in the survey mapping based on the given question text.
    Uses TF-IDF vectorization and cosine similarity for matching.
    """
    # Transform the input question text to a vector
    input_vector = vectorizer.transform([question_text])

    # Calculate cosine similarity between the input text and all survey questions
    similarity_scores = cosine_similarity(input_vector, survey_vectors).flatten()

    # Find the index of the most similar question
    max_similarity_index = np.argmax(similarity_scores)
    max_similarity_score = similarity_scores[max_similarity_index]

    # Check if the highest similarity score is above the threshold
    if max_similarity_score >= similarity_threshold:
        # Get the corresponding question ID from the survey mapping
        question_id = survey_mapping.iloc[max_similarity_index]['English Question ID']
        print(f"Matched question with similarity score: {max_similarity_score}")
        return question_id
    else:
        print("No matching question found above the similarity threshold.")
        return None


def get_question_text_and_options(question_id):
    # Find the question in the survey mapping
    question_row = survey_mapping[survey_mapping['English Question ID'] == question_id]
    if not question_row.empty:
        question_text = question_row.iloc[0]['English Question Text']
        response_options = ast.literal_eval(question_row.iloc[0]['English Response Options'])
        return question_text, response_options
    else:
        return None, None
    
def get_survey_results(prompt, question_id):
    # Parse the input to extract features
    features = parse_input(prompt)
    print(f"Extracted features: {features}")

    # Match personas based on the extracted features
    matching_personas = match_personas(features)
    if matching_personas.empty:
        print("No matching personas found.")
        return None

    # Calculate the survey response distribution for the given question
    response_distribution, matching_count = calculate_response_distribution(question_id, matching_personas)
    if response_distribution is not None:
        # Get the question text and options
        question_text, response_options = get_question_text_and_options(question_id)
        if question_text and response_options:
            print(f"Question: {question_text}")
            print("Response options:")
            for idx, option in enumerate(response_options, start=1):
                print(f"{idx}. {option}")

        print(f"\nSurvey results for question {question_id} based on the matching personas:")
        print(f"Number of persons matching the criteria: {matching_count}")
        return response_distribution
    else:
        print(f"The question {question_id} does not exist in the survey data.")
        return None

def get_survey_results_with_text(prompt, question_text):
    # Find the question ID based on the provided question text
    print(question_text)
    question_id = find_question_id_by_text(question_text)
    if not question_id:
        print("Question not found.")
        return None
    print(question_id)
    # Parse the input to extract features
    features = parse_input(prompt)
    print(f"Extracted features: {features}")

    # Match personas based on the extracted features
    matching_personas = match_personas(features)
    if matching_personas.empty:
        print("No matching personas found.")
        return None

    # Calculate the survey response distribution for the given question
    response_distribution, matching_count = calculate_response_distribution(question_id, matching_personas)
    if response_distribution is not None:
        # Get the question text and options
        question_text, response_options = get_question_text_and_options(question_id)
        if question_text and response_options:
            print(f"Question: {question_text}")
            print("Response options:")
            for idx, option in enumerate(response_options, start=1):
                print(f"{idx}. {option}")

        print(f"\nSurvey results for question {question_id} based on the matching personas:")
        print(f"Number of persons matching the criteria: {matching_count}")
        return response_distribution
    else:
        print(f"The question {question_id} does not exist in the survey data.")
        return None

# Example usage
prompt = "Ayşe, 35 years old female, married, with 2 children, has a primary education, and is Housewife not otherwise employed"
prompt1 = "Ayşe, 35 years old "

question_text = "For each of the following, indicate how important it is in your life. Would you say it is very important, rather important, not very important or not important at all? Family"
question_text2 = "For each of the following, indicate how important it is in your life. Would you say it is very important, rather important, not very important or not important at all? Friends"
results_text = get_survey_results_with_text(prompt1, question_text)    
if results_text is not None:
    print(results_text)

print("-----------------------")
# Example usage
prompt = "Ayşe, 35 years old female, married, with 2 children, has a primary education, and is Housewife not otherwise employed"
prompt1 = "Ayşe"

question_id = 'Q2'  # Replace with the actual question ID you want to analyze
results = get_survey_results(prompt, question_id)
