import pandas as pd
import ast  # For safely evaluating string representations of lists

survey_mapping = pd.read_csv('results/survey_question_mapping.csv')

survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')

final_output = []

def categorize_age(age):
    if age <= 29:
        return 'Up to 29'
    elif 30 <= age <= 49:
        return '30-49'
    else:
        return '50 and more'

# Categorize the age column into groups
survey_data['Age Group'] = survey_data['Q262'].apply(categorize_age)

# Map gender codes to labels
gender_map = {1: 'Male', 2: 'Female'}
survey_data['Gender'] = survey_data['Q260'].map(gender_map)

# Map marital status codes to labels
marital_status_map = {1: 'Married', 6: 'Single', 3: 'Divorced'}
survey_data['Marital Status'] = survey_data['Q273'].map(marital_status_map)

# Map number of children
survey_data['Children Group'] = survey_data['Q274'].apply(
    lambda x: '4 or more children' if x >= 4 else f"{int(x)} child{'ren' if x > 1 else ''}"
)

# Map education levels for Q275
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
survey_data['Education Level'] = survey_data['Q275'].map(education_level_map)

# Map employment status for Q279
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
survey_data['Employment Status'] = survey_data['Q279'].map(employment_status_map)

# Map social class for Q287
social_class_map = {
    1: 'Upper class',
    2: 'Upper middle class',
    3: 'Lower middle class',
    4: 'Working class',
    5: 'Lower class'
}
survey_data['Social Class'] = survey_data['Q287'].map(social_class_map)

for index, row in survey_mapping.iterrows():
    try:
        # Extract the English Question ID and the English Response Options
        english_question_id = row['English Question ID']
        english_question_text = row['English Question Text']

        # Handle NaN values or invalid rows in the 'English Response Options'
        if pd.isna(row['English Response Options']):
            print(f"Skipping row {index} due to NaN in English Response Options")
            continue

        # Safely evaluate the string to a list using ast.literal_eval
        try:
            english_options = ast.literal_eval(row['English Response Options'])
        except (SyntaxError, ValueError) as e:
            print(f"Skipping row {index} due to invalid string format in English Response Options: {e}")
            continue

        # Find the matching column in the survey data
        column_name = f"{english_question_id}"

        # Check if this column exists in the survey data
        if column_name in survey_data.columns:
            # Get the unique response codes for this question from the data
            unique_codes = survey_data[column_name].dropna().unique()
            unique_codes = sorted(unique_codes)

            # Create a mapping between response codes and options
            response_code_to_option = {i + 1: option for i, option in enumerate(english_options)}

            # For Q275, Q279, and Q287, use predefined mappings
            if english_question_id == 'Q275':
                response_code_to_option = education_level_map
            elif english_question_id == 'Q279':
                response_code_to_option = employment_status_map
            elif english_question_id == 'Q287':
                response_code_to_option = social_class_map

            total_responses = survey_data[column_name].value_counts(normalize=True) * 100

            # Extract demographic-specific responses
            def get_demographic_responses(demo_column, demo_value):
                return survey_data[survey_data[demo_column] == demo_value][column_name].value_counts(normalize=True) * 100

            # Gender
            male_responses = get_demographic_responses('Gender', 'Male')
            female_responses = get_demographic_responses('Gender', 'Female')

            # Age groups
            up_to_29_responses = get_demographic_responses('Age Group', 'Up to 29')
            age_30_49_responses = get_demographic_responses('Age Group', '30-49')
            age_50_and_more_responses = get_demographic_responses('Age Group', '50 and more')

            # Marital status
            married_responses = get_demographic_responses('Marital Status', 'Married')
            single_responses = get_demographic_responses('Marital Status', 'Single')
            divorced_responses = get_demographic_responses('Marital Status', 'Divorced')

            # Number of children
            no_children_responses = get_demographic_responses('Children Group', '0 child')
            one_child_responses = get_demographic_responses('Children Group', '1 child')
            two_children_responses = get_demographic_responses('Children Group', '2 children')
            three_children_responses = get_demographic_responses('Children Group', '3 children')
            four_or_more_children_responses = get_demographic_responses('Children Group', '4 or more children')

            # Education levels
            education_levels_of_interest = [
                'Early childhood education (ISCED 0) / no education', 
                'Primary education (ISCED 1)', 
                'Lower secondary education (ISCED 2)', 
                'Upper secondary education (ISCED 3)', 
                'Bachelor or equivalent (ISCED 6)'
            ]
            education_responses = {}
            for level in education_levels_of_interest:
                education_responses[level] = get_demographic_responses('Education Level', level)

            # Employment status
            employment_statuses = [
                'Full-time employee (30 hours a week or more)',
                'Part-time employee (less than 30 hours a week)',
                'Self-employed',
                'Retired/pensioned',
                'Housewife not otherwise employed',
                'Student',
                'Unemployed',
                'Other (write in)'
            ]
            employment_responses = {}
            for status in employment_statuses:
                employment_responses[status] = get_demographic_responses('Employment Status', status)

            # Social Class
            social_classes = [
                'Upper class', 
                'Upper middle class', 
                'Lower middle class', 
                'Working class', 
                'Lower class'
            ]
            social_class_responses = {}
            for social_class in social_classes:
                social_class_responses[social_class] = get_demographic_responses('Social Class', social_class)

            # Loop over the response codes and options
            for code, option in response_code_to_option.items():
                # General percentage
                total_percentage = total_responses.get(code, 0)

                # Gender percentages
                male_percentage = male_responses.get(code, 0)
                female_percentage = female_responses.get(code, 0)

                # Age group percentages
                up_to_29_percentage = up_to_29_responses.get(code, 0)
                age_30_49_percentage = age_30_49_responses.get(code, 0)
                age_50_and_more_percentage = age_50_and_more_responses.get(code, 0)

                # Marital status percentages
                married_percentage = married_responses.get(code, 0)
                single_percentage = single_responses.get(code, 0)
                divorced_percentage = divorced_responses.get(code, 0)

                # Number of children percentages
                no_children_percentage = no_children_responses.get(code, 0)
                one_child_percentage = one_child_responses.get(code, 0)
                two_children_percentage = two_children_responses.get(code, 0)
                three_children_percentage = three_children_responses.get(code, 0)
                four_or_more_children_percentage = four_or_more_children_responses.get(code, 0)

                # Education level percentages
                no_education_percentage = education_responses['Early childhood education (ISCED 0) / no education'].get(code, 0)
                primary_education_percentage = education_responses['Primary education (ISCED 1)'].get(code, 0)
                lower_secondary_percentage = education_responses['Lower secondary education (ISCED 2)'].get(code, 0)
                upper_secondary_percentage = education_responses['Upper secondary education (ISCED 3)'].get(code, 0)
                bachelor_percentage = education_responses['Bachelor or equivalent (ISCED 6)'].get(code, 0)

                # Employment status percentages
                full_time_percentage = employment_responses['Full-time employee (30 hours a week or more)'].get(code, 0)
                part_time_percentage = employment_responses['Part-time employee (less than 30 hours a week)'].get(code, 0)
                self_employed_percentage = employment_responses['Self-employed'].get(code, 0)
                retired_percentage = employment_responses['Retired/pensioned'].get(code, 0)
                housewife_percentage = employment_responses['Housewife not otherwise employed'].get(code, 0)
                student_percentage = employment_responses['Student'].get(code, 0)
                unemployed_percentage = employment_responses['Unemployed'].get(code, 0)
                other_percentage = employment_responses['Other (write in)'].get(code, 0)

                # Social class percentages
                upper_class_percentage = social_class_responses['Upper class'].get(code, 0)
                upper_middle_class_percentage = social_class_responses['Upper middle class'].get(code, 0)
                lower_middle_class_percentage = social_class_responses['Lower middle class'].get(code, 0)
                working_class_percentage = social_class_responses['Working class'].get(code, 0)
                lower_class_percentage = social_class_responses['Lower class'].get(code, 0)

                # Prepare the final output dictionary
                final_output.append({
                    'Survey Question ID': english_question_id,
                    'Survey Question English Text': english_question_text,
                    'Response Option': option,
                    'Total': f"%{round(total_percentage, 2)}",
                    'Male': f"%{round(male_percentage, 2)}",
                    'Female': f"%{round(female_percentage, 2)}",
                    'Up to 29': f"%{round(up_to_29_percentage, 2)}",
                    '30-49': f"%{round(age_30_49_percentage, 2)}",
                    '50 and more': f"%{round(age_50_and_more_percentage, 2)}",
                    'Married': f"%{round(married_percentage, 2)}",
                    'Single': f"%{round(single_percentage, 2)}",
                    'Divorced': f"%{round(divorced_percentage, 2)}",
                    'No children': f"%{round(no_children_percentage, 2)}",
                    '1 child': f"%{round(one_child_percentage, 2)}",
                    '2 children': f"%{round(two_children_percentage, 2)}",
                    '3 children': f"%{round(three_children_percentage, 2)}",
                    '4 or more children': f"%{round(four_or_more_children_percentage, 2)}",
                    'No education': f"%{round(no_education_percentage, 2)}",
                    'Primary education': f"%{round(primary_education_percentage, 2)}",
                    'Lower secondary': f"%{round(lower_secondary_percentage, 2)}",
                    'Upper secondary': f"%{round(upper_secondary_percentage, 2)}",
                    'Bachelor': f"%{round(bachelor_percentage, 2)}",
                    'Full-time employee': f"%{round(full_time_percentage, 2)}",
                    'Part-time employee': f"%{round(part_time_percentage, 2)}",
                    'Self-employed': f"%{round(self_employed_percentage, 2)}",
                    'Retired/pensioned': f"%{round(retired_percentage, 2)}",
                    'Housewife': f"%{round(housewife_percentage, 2)}",
                    'Student': f"%{round(student_percentage, 2)}",
                    'Unemployed': f"%{round(unemployed_percentage, 2)}",
                    'Other': f"%{round(other_percentage, 2)}",
                    'Upper class': f"%{round(upper_class_percentage, 2)}",
                    'Upper middle class': f"%{round(upper_middle_class_percentage, 2)}",
                    'Lower middle class': f"%{round(lower_middle_class_percentage, 2)}",
                    'Working class': f"%{round(working_class_percentage, 2)}",
                    'Lower class': f"%{round(lower_class_percentage, 2)}"
                })

    except Exception as e:
        print(f"Error processing row {index}: {e}")

if final_output:
    final_df = pd.DataFrame(final_output)
    final_df.to_csv('results/final_survey_results_without_persona.csv', index=False)
else:
    print("No valid data to output.")