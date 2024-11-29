import ast  # Import the ast module for safely evaluating strings
import pandas as pd

# Load the data
persona_df = pd.read_csv('results/persona_counts_with_prompts_tr.csv')
survey_mapping = pd.read_csv('results/survey_question_mapping.csv')

# Filter questions up to Q259
survey_mapping = survey_mapping[survey_mapping['Turkish Question ID'].str.startswith('Q')]
survey_mapping = survey_mapping[
    survey_mapping['Turkish Question ID'].str.extract(r'(\d+)')[0].astype(int) <= 259
]


def create_prompt(persona_prompt, question_text, response_options):
    try:
        # Attempt to parse the response options
        response_options = ast.literal_eval(response_options)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing response options: {response_options} - {e}")
        response_options = ["(Hatalı seçenekler)"]  # Fallback option if parsing fails

    # Format response options as a numbered list
    formatted_options = "\n".join([f"{i + 1}. {option}" for i, option in enumerate(response_options)])

    # Construct the prompt
    prompt = (
        f"{persona_prompt}\n\n"
        f"Bu kişi aşağıdaki soruya ne cevap verir?\n"
        f"{question_text}\n\n"
        f"Cevap seçenekleri:\n{formatted_options}\n\n"
        f"Yalnızca bir yanıt seçin (1, 2, 3, vb.):"
    )
    return prompt

# Prepare to write to CSV incrementally
output_csv = 'results/final_prompts.csv'
with open(output_csv, 'w', encoding='utf-8') as f_out:
    #I removed Final Prompt for now because we create prompt in collab.
    f_out.write('Persona,Question,Response Options\n') 
    #f_out.write('Persona,Question,Response Options,Final Prompt\n')
    
    # Process personas and questions in chunks
    persona_chunk_size = 100
    question_chunk_size = 50

    num_personas = len(persona_df)
    num_questions = len(survey_mapping)

    for persona_start in range(0, num_personas, persona_chunk_size):
        persona_end = min(persona_start + persona_chunk_size, num_personas)
        persona_chunk = persona_df.iloc[persona_start:persona_end].copy()
        persona_chunk['key'] = 1

        for question_start in range(0, num_questions, question_chunk_size):
            question_end = min(question_start + question_chunk_size, num_questions)
            question_chunk = survey_mapping.iloc[question_start:question_end].copy()
            question_chunk['key'] = 1

            # Merge persona chunk with question chunk
            combined_chunk = pd.merge(persona_chunk, question_chunk, on='key')

            # Create prompts with Turkish response options included
            """
            combined_chunk['Final Prompt'] = combined_chunk.apply(
                lambda row: create_prompt(
                    row['Prompt'],
                    row['Turkish Question Text'],
                    row['Turkish Response Options']
                ), axis=1)

            # Add persona, question, and response options to the output
            combined_chunk[['Prompt', 'Turkish Question Text', 'Turkish Response Options', 'Final Prompt']].to_csv(
                f_out, header=False, index=False, mode='a', encoding='utf-8')
            """
            combined_chunk[['Prompt', 'Turkish Question Text', 'Turkish Response Options']].to_csv(
                f_out, header=False, index=False, mode='a', encoding='utf-8')


            #print(f"Processed personas {persona_start}-{persona_end}, questions {question_start}-{question_end}")

print("Final prompts generated and saved to 'final_prompts.csv'")