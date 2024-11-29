import pandas as pd
from itertools import product

survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')

# Gender
gender_map = {1: 'Male', 2: 'Female'}

# Age groups
def categorize_age(age):
    if age <= 29:
        return 'Up to 29'
    elif 30 <= age <= 49:
        return '30-49'
    else:
        return '50 and more'

# Marital status
marital_status_map = {1: 'Married', 6: 'Single', 3: 'Divorced'}

# Number of children
def categorize_children(children):
    if pd.isna(children):
        return None
    return '4 or more children' if children >= 4 else f"{int(children)} child{'ren' if children > 1 else ''}"

# Education levels
education_level_map = {
    0: 'Early childhood education (ISCED 0) / no education',
    1: 'Primary education (ISCED 1)',
    2: 'Lower secondary education (ISCED 2)',
    3: 'Upper secondary education (ISCED 3)',
    4: 'Post-secondary non-tertiary education (ISCED 4)',
    5: 'Short-cycle tertiary education (ISCED 5)',
    6: 'Bachelor or equivalent (ISCED 6)',
    7: 'Master or equivalent (ISCED 7)',
    8: 'Doctoral or equivalent (ISCED 8)'
}

# Employment status
employment_status_map = {
    1: 'Full-time employee (30 hours a week or more)',
    2: 'Part-time employee (less than 30 hours a week)',
    3: 'Self-employed',
    4: 'Retired/pensioned',
    5: 'Housewife not otherwise employed',
    6: 'Student',
    7: 'Unemployed',
    8: 'Other (write in)',
    -1: 'No answer',
    -2: 'Don’t know',
    -3: 'Not applicable',
    -4: 'Missing; Unknown'
}

# Social class
social_class_map = {
    1: 'Upper class',
    2: 'Upper middle class',
    3: 'Lower middle class',
    4: 'Working class',
    5: 'Lower class'
}

survey_data['Gender'] = survey_data['Q260'].map(gender_map)
survey_data['Age Group'] = survey_data['Q262'].apply(categorize_age)
survey_data['Marital Status'] = survey_data['Q273'].map(marital_status_map)
survey_data['Children Group'] = survey_data['Q274'].apply(categorize_children)
survey_data['Education Level'] = survey_data['Q275'].map(education_level_map)
survey_data['Employment Status'] = survey_data['Q279'].map(employment_status_map)
survey_data['Social Class'] = survey_data['Q287'].map(social_class_map)

survey_data_filtered = survey_data.dropna(subset=[
    'Gender', 'Age Group', 'Marital Status', 'Children Group',
    'Education Level', 'Employment Status', 'Social Class'
])

gender_choices = ['Male', 'Female']
age_choices = ['Up to 29', '30-49', '50 and more']
marital_status_choices = ['Married', 'Single', 'Divorced']
children_choices = ['0 children', '1 child', '2 children', '3 children', '4 or more children']
education_choices = [
    'Early childhood education (ISCED 0) / no education', 
    'Primary education (ISCED 1)', 
    'Lower secondary education (ISCED 2)', 
    'Upper secondary education (ISCED 3)', 
    'Post-secondary non-tertiary education (ISCED 4)', 
    'Short-cycle tertiary education (ISCED 5)', 
    'Bachelor or equivalent (ISCED 6)', 
    'Master or equivalent (ISCED 7)', 
    'Doctoral or equivalent (ISCED 8)'
]
employment_choices = [
    'Full-time employee (30 hours a week or more)',
    'Part-time employee (less than 30 hours a week)',
    'Self-employed',
    'Retired/pensioned',
    'Housewife not otherwise employed',
    'Student',
    'Unemployed',
    'Other (write in)'
]
social_class_choices = [
    'Upper class', 
    'Upper middle class', 
    'Lower middle class', 
    'Working class', 
    'Lower class'
]

all_personas = list(product(
    gender_choices, age_choices, marital_status_choices, children_choices,
    education_choices, employment_choices, social_class_choices
))

persona_counts = []
total_number = 0

for persona in all_personas:
    gender, age_group, marital_status, children_group, education_level, employment_status, social_class = persona
    # Filter the survey data to match the persona
    matching_records = survey_data_filtered[
        (survey_data_filtered['Gender'] == gender) &
        (survey_data_filtered['Age Group'] == age_group) &
        (survey_data_filtered['Marital Status'] == marital_status) &
        (survey_data_filtered['Children Group'] == children_group) &
        (survey_data_filtered['Education Level'] == education_level) &
        (survey_data_filtered['Employment Status'] == employment_status) &
        (survey_data_filtered['Social Class'] == social_class)
    ]
    # Count the number of matches
    count = len(matching_records)
    total_number += count
    persona_counts.append({
        'Gender': gender,
        'Age Group': age_group,
        'Marital Status': marital_status,
        'Children Group': children_group,
        'Education Level': education_level,
        'Employment Status': employment_status,
        'Social Class': social_class,
        'Count': count
    })

print(f"Total matched personas: {total_number}")

persona_df = pd.DataFrame(persona_counts)

important_personas = persona_df[persona_df['Count'] > 0].sort_values(by='Count', ascending=False)

# Save the DataFrame with all personas and their corresponding counts to a CSV file
important_personas.to_csv('results/rest/important_persona_counts_en.csv', index=False)
important_personas