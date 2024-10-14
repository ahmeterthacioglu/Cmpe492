import pandas as pd
import ast  # For safely evaluating string representations of lists

# Step 1: Load the survey mapping file
survey_mapping = pd.read_csv('results/survey_question_mapping.csv')

# Step 2: Load the survey data file
survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')

# Step 3: Prepare an empty list for storing the final output rows
final_output = []

# Step 4: Define a function to categorize ages into groups
def categorize_age(age):
    if age <= 29:
        return 'Up to 29'
    elif 30 <= age <= 49:
        return '30-49'
    else:
        return '50 and more'

# Step 5: Loop through each row in the survey mapping file
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
        
        # Add the additional options (Don't know, No answer, etc.)
        additional_responses = ['Don’t know', 'No answer', 'Other missing; Multiple answers Mail (EVS)']
        english_options.extend(additional_responses)
        
        # Find the matching column in the survey data based on the question ID (assumed to be "Q" followed by the number)
        column_name = f"{english_question_id}"
        
        # Check if this column exists in the survey data and ensure relevant columns (gender, age, marital status, children) exist
        if column_name in survey_data.columns and 'Q260' in survey_data.columns and 'Q262' in survey_data.columns and 'Q273' in survey_data.columns and 'Q274' in survey_data.columns:
            # Step 6: Categorize the age column into groups (Up to 29, 30-49, 50 and more)
            survey_data['Age Group'] = survey_data['Q262'].apply(categorize_age)
            
            # Step 7: Extract the overall percentages from the corresponding survey column
            total_responses = survey_data[column_name].value_counts(normalize=True) * 100
            
            # Extract gender data from Q260 (1 for Male, 2 for Female)
            male_responses = survey_data[survey_data['Q260'] == 1][column_name].value_counts(normalize=True) * 100
            female_responses = survey_data[survey_data['Q260'] == 2][column_name].value_counts(normalize=True) * 100
            
            # Extract age group data
            up_to_29_responses = survey_data[survey_data['Age Group'] == 'Up to 29'][column_name].value_counts(normalize=True) * 100
            age_30_49_responses = survey_data[survey_data['Age Group'] == '30-49'][column_name].value_counts(normalize=True) * 100
            age_50_and_more_responses = survey_data[survey_data['Age Group'] == '50 and more'][column_name].value_counts(normalize=True) * 100
            
            # Extract marital status from Q273 (e.g., 1 = Married, 6 = Single, etc.)
            married_responses = survey_data[survey_data['Q273'] == 1][column_name].value_counts(normalize=True) * 100
            single_responses = survey_data[survey_data['Q273'] == 6][column_name].value_counts(normalize=True) * 100
            divorced_responses = survey_data[survey_data['Q273'] == 3][column_name].value_counts(normalize=True) * 100
            
            # Extract number of children from Q274 (0 = No children, 1 = 1 child, etc.)
            no_children_responses = survey_data[survey_data['Q274'] == 0][column_name].value_counts(normalize=True) * 100
            one_child_responses = survey_data[survey_data['Q274'] == 1][column_name].value_counts(normalize=True) * 100
            two_children_responses = survey_data[survey_data['Q274'] == 2][column_name].value_counts(normalize=True) * 100
            three_children_responses = survey_data[survey_data['Q274'] == 3][column_name].value_counts(normalize=True) * 100
            four_or_more_children_responses = survey_data[survey_data['Q274'] >= 4][column_name].value_counts(normalize=True) * 100
            
            # Create a list to store the response options with overall and demographic-specific percentages
            for idx, option in enumerate(english_options, start=1):
                # General percentage
                total_percentage = total_responses.get(idx, 0)
                
                # Gender percentages
                male_percentage = male_responses.get(idx, 0)
                female_percentage = female_responses.get(idx, 0)
                
                # Age group percentages
                up_to_29_percentage = up_to_29_responses.get(idx, 0)
                age_30_49_percentage = age_30_49_responses.get(idx, 0)
                age_50_and_more_percentage = age_50_and_more_responses.get(idx, 0)
                
                # Marital status percentages
                married_percentage = married_responses.get(idx, 0)
                single_percentage = single_responses.get(idx, 0)
                divorced_percentage = divorced_responses.get(idx, 0)
                
                # Number of children percentages
                no_children_percentage = no_children_responses.get(idx, 0)
                one_child_percentage = one_child_responses.get(idx, 0)
                two_children_percentage = two_children_responses.get(idx, 0)
                three_children_percentage = three_children_responses.get(idx, 0)
                four_or_more_children_percentage = four_or_more_children_responses.get(idx, 0)
                
                # Step 8: Append the data to the final output (one row per response option)
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
                    '4 or more children': f"%{round(four_or_more_children_percentage, 2)}"
                })
    
    except Exception as e:
        print(f"Error processing row {index}: {e}")

# Step 9: Convert the final output to a DataFrame and write to a new CSV file
if final_output:
    final_df = pd.DataFrame(final_output)
    final_df.to_csv('final_survey_results_with_gender_age_marital_children.csv', index=False)
else:
    print("No valid data to output.")