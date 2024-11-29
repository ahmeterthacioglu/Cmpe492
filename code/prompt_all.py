import pandas as pd
import numpy as np
import ast
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the processed persona data and survey data
persona_df = pd.read_csv('results/persona_counts_with_prompts_tr.csv', dtype={'Count': int})
survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')
survey_mapping = pd.read_csv('results/survey_question_mapping.csv')  

def normalize_text(text):
    replacements = {
        'Ü': 'U', 'ü': 'u',
        'Ş': 'S', 'ş': 's',
        'Ç': 'C', 'ç': 'c',
        'Ğ': 'G', 'ğ': 'g',
        'Ö': 'O', 'ö': 'o',
        'İ': 'I', 'ı': 'i',
    }
    # Normalize text to NFD to decompose characters
    text = unicodedata.normalize('NFD', str(text))
    # Replace special Turkish characters
    for src, target in replacements.items():
        text = text.replace(src, target)
    # Remove any remaining combining marks
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Convert to lowercase using casefold
    text = text.casefold()
    return text


# Define mappings at the global scope
gender_map = {1: 'Erkek', 2: 'Kadın'}

def marital_status_map(index):
    if index in [1, 2]:
        return 'Evli'
    elif index in [3, 4, 5, 6]:
        return 'Bekâr'
    return None

def categorize_children(children):
    if pd.isna(children):
        return None
    elif children >= 1:
        return 'Çocuk sahibi'
    else:
        return 'Çocuğu yok'

def education_level_map(index):
    if index == 792001:
        return 'Hiç okula gitmemiş'
    elif index == 792014:
        return 'İlkokul terk'
    elif index in [792015, 792016]:
        return 'İlkokul mezunu'
    elif index in [792017, 792018]:
        return 'Ortaokul mezunu'
    elif index in [792019, 792020]:
        return 'Lise mezunu'
    elif index == 792021:
        return 'Üniversite mezunu'
    elif index in [792023, 792024]:
        return 'Şu anda öğrenci'
    return None

employment_status_map = {
    1: 'Ücretli ve tam zamanlı çalışan',
    2: 'Ücretli ve yarı zamanlı çalışan',
    3: 'Kendi işinin sahibi',
    4: 'Emekli',
    5: 'Ev kadını',
    6: 'Öğrenci',
    7: 'İşsiz/iş arayan'
}

social_class_map = {
    1: 'Üst sınıf',
    2: 'Orta sınıfın üst kısmında',
    3: 'Orta sınıfın alt kısmında',
    4: 'Çalışan, işçi, emekçi sınıfı',
    5: 'Alt sınıf'
}

# Settlement type mapping
settlement_type_map = {
    1: 'Kent merkezinde',
    2: 'Kırsal alanda'
}

# Region mapping
region_map = {
    792101: 'İstanbul',
    792102: 'Tekirdağ, Edirne, Kırklareli',
    792103: 'Balıkesir, Çanakkale',
    792104: 'İzmir',
    792105: 'Aydın, Denizli, Muğla',
    792106: 'Manisa, Afyon, Kütahya, Uşak',
    792107: 'Bursa, Eskişehir, Bilecik',
    792108: 'Kocaeli, Sakarya, Düzce, Bolu, Yalova',
    792109: 'Ankara',
    792110: 'Konya, Karaman',
    792111: 'Antalya, Isparta, Burdur',
    792112: 'Adana, Mersin',
    792113: 'Hatay, Kahramanmaraş, Osmaniye',
    792114: 'Kırıkkale, Aksaray, Niğde, Kırşehir, Nevşehir',
    792115: 'Kayseri, Sivas, Yozgat',
    792116: 'Zonguldak, Karabük, Bartın',
    792117: 'Kastamonu, Çankırı, Sinop',
    792118: 'Samsun, Tokat, Çorum, Amasya',
    792119: 'Trabzon, Ordu, Giresun, Rize, Artvin, Gümüşhane',
    792120: 'Erzurum, Erzincan, Bayburt',
    792121: 'Ağrı, Kars, Iğdır, Ardahan',
    792122: 'Malatya, Elazığ, Bingöl, Tunceli',
    792123: 'Van, Muş, Bitlis, Hakkari',
    792124: 'Gaziantep, Adıyaman, Kilis',
    792125: 'Şanlıurfa, Diyarbakır',
    792126: 'Mardin, Batman, Şırnak, Siirt'
}

def categorize_age(age):
    if age <= 29:
        return '30 yaşından küçük'
    elif 30 <= age <= 49:
        return '30-50 yaş arası'
    else:
        return '50 yaşında veya daha yaşlı'

def preprocess_survey_data(survey_data):
    # Apply mappings
    survey_data['Cinsiyet'] = survey_data['Q260'].map(gender_map)
    survey_data['Yaş Grubu'] = survey_data['Q262'].apply(categorize_age)
    survey_data['Medeni Durum'] = survey_data['Q273'].apply(marital_status_map)
    survey_data['Çocuk Sahipliği'] = survey_data['Q274'].apply(categorize_children)
    survey_data['Eğitim Düzeyi'] = survey_data['Q275A'].apply(education_level_map)
    survey_data['İş Durumu'] = survey_data['Q279'].map(employment_status_map)
    survey_data['Sosyal Sınıf'] = survey_data['Q287'].map(social_class_map)
    survey_data['Yerleşim Yeri'] = survey_data['H_URBRURAL'].map(settlement_type_map)
    survey_data['Bölge'] = survey_data['N_REGION_WVS'].map(region_map)
    
    return survey_data

# Preprocess the survey data
survey_data = preprocess_survey_data(survey_data)

# Standardize 'Bölge' in persona_df
persona_df['Bölge'] = persona_df['Bölge'].str.replace(r'^TR\d+: ', '', regex=True).str.strip()
# Standardize 'İş Durumu' in persona_df
persona_df['İş Durumu'] = persona_df['İş Durumu'].str.replace(r'\s*\(.*\)', '', regex=True).str.strip()
# Strip whitespace from string columns in persona_df
string_columns = ['Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
                  'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri', 'Bölge']
for col in string_columns:
    persona_df[col] = persona_df[col].astype(str).str.strip()

def parse_input(prompt):
    import re
    
    # Remove text in parentheses
    prompt = re.sub(r'\(.*?\)', '', str(prompt))
    
    # Normalize the prompt
    prompt_norm = normalize_text(prompt)
    
    # Initialize features as None
    gender = None
    age_group = None
    marital_status = None
    children = None
    education = None
    employment = None
    social_class = None
    settlement_size = None
    region = None

    # Extract gender
    if 'kadin' in prompt_norm:
        gender = 'Kadın'
    elif 'erkek' in prompt_norm:
        gender = 'Erkek'

    # Extract age
    age_match = re.search(r'(\d{1,3}) yasinda', prompt_norm)
    if age_match:
        age = int(age_match.group(1))
        age_group = categorize_age(age)

    # Extract marital status
    if 'evli' in prompt_norm:
        marital_status = 'Evli'
    elif 'bekar' in prompt_norm:
        marital_status = 'Bekâr'

    # Extract children
    if 'cocuk sahibi' in prompt_norm:
        children = 'Çocuk sahibi'
    elif 'cocugu yok' in prompt_norm:
        children = 'Çocuğu yok'

    # Extract education level
    education_levels = [
        'Hiç okula gitmemiş', 'İlkokul terk', 'İlkokul mezunu', 'Ortaokul mezunu',
        'Lise mezunu', 'Üniversite mezunu', 'Şu anda öğrenci'
    ]
    education_levels_norm = [normalize_text(level) for level in education_levels]
    for level_norm, level_original in zip(education_levels_norm, education_levels):
        if level_norm in prompt_norm:
            education = level_original
            break

    # Extract employment status
    employment_options = list(employment_status_map.values())
    employment_options_norm = [normalize_text(option) for option in employment_options]
    for option_norm, option_original in zip(employment_options_norm, employment_options):
        if option_norm in prompt_norm:
            employment = option_original
            break

    # Extract social class
    social_classes = list(social_class_map.values())
    social_classes_norm = [normalize_text(s_class) for s_class in social_classes]
    for s_class_norm, s_class_original in zip(social_classes_norm, social_classes):
        if s_class_norm in prompt_norm:
            social_class = s_class_original
            break

    # Extract settlement type ('Yerleşim Yeri')
    if 'kent merkezinde' in prompt_norm:
        settlement_size = 'Kent merkezinde'
    elif 'kirsal alanda' in prompt_norm:
        settlement_size = 'Kırsal alanda'

    # Extract region ('Bölge')
    regions = []
    for region_list in region_map.values():
        regions.extend([reg.strip() for reg in region_list.split(',')])
    regions_norm = [normalize_text(reg) for reg in regions]
    for reg_norm, reg_original in zip(regions_norm, regions):
        if reg_norm in prompt_norm:
            region = reg_original
            break

    # Return extracted features
    return {
        'Cinsiyet': gender,
        'Yaş Grubu': age_group,
        'Medeni Durum': marital_status,
        'Çocuk Sahipliği': children,
        'Eğitim Düzeyi': education,
        'İş Durumu': employment,
        'Sosyal Sınıf': social_class,
        'Yerleşim Yeri': settlement_size,
        'Bölge': region
    }

def match_personas(features):
    matching_personas = survey_data.copy()
    
    # Ensure 'Bölge' column is string type and handle NaN values
    matching_personas['Bölge'] = matching_personas['Bölge'].astype(str).fillna('')
    
    for feature, value in features.items():
        if value is not None and feature in matching_personas.columns:
            if feature == 'Bölge':
                # Handle 'Bölge' matching with partial matching
                matching_personas = matching_personas[matching_personas['Bölge'].str.contains(value, na=False)]
            else:
                matching_personas = matching_personas[matching_personas[feature] == value]
    return matching_personas

def calculate_response_distribution(question_id, matching_personas):
    # Proceed if data is available
    if not matching_personas.empty and question_id in matching_personas.columns:
        # Remove rows with missing responses to the question
        filtered_data = matching_personas[matching_personas[question_id].notna()]
        # Calculate response distribution
        response_counts = filtered_data[question_id].value_counts()
        total_responses = response_counts.sum()
        response_distribution = (response_counts / total_responses) * 100
        response_distribution = response_distribution.sort_index()
        matching_count = len(filtered_data)
        return response_distribution, matching_count
    else:
        return pd.Series(dtype=float), 0

def safe_literal_eval(value):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(value)
    except Exception:
        # Attempt to fix common issues
        value_fixed = value.strip()
        if not value_fixed.startswith('['):
            value_fixed = '[' + value_fixed
        if not value_fixed.endswith(']'):
            value_fixed = value_fixed + ']'
        # Ensure that all elements are properly quoted
        elements = re.split(r",\s*(?![^[]*\])", value_fixed.strip('[]'))
        elements = [elem.strip().strip("'\"") for elem in elements]
        elements = [f"'{elem}'" for elem in elements]
        value_fixed = '[' + ', '.join(elements) + ']'
        try:
            return ast.literal_eval(value_fixed)
        except Exception as e:
            print(f"Failed to parse after fixing: {value_fixed}")
            return []

def process_all_prompts_and_questions(persona_df):
     # Filter for questions up to Q259
    survey_mapping_filtered = survey_mapping[survey_mapping['Turkish Question ID'].str.startswith('Q')]
    survey_mapping_filtered = survey_mapping_filtered[
        survey_mapping_filtered['Turkish Question ID'].str.extract(r'(\d+)')[0].astype(int) <= 259
    ]

    # Prepare a list of all question IDs and texts
    question_ids = survey_mapping_filtered['Turkish Question ID'].tolist()
    question_texts = survey_mapping_filtered['Turkish Question Text'].tolist()
    response_options_list = survey_mapping_filtered['Turkish Response Options'].apply(safe_literal_eval).tolist()

    results = []

    # Pre-extract features for all prompts
    persona_df['Extracted Features'] = persona_df['Prompt'].apply(parse_input)

    for index, row in persona_df.iterrows():
        prompt = row['Prompt']
        features = row['Extracted Features']
        count = row['Count']  # Get the Count from the persona DataFrame

        # Match personas based on the extracted features
        matching_personas = match_personas(features)
        matching_count = len(matching_personas)

        for q_id, q_text, response_options in zip(question_ids, question_texts, response_options_list):
            # Calculate the survey response distribution for the given question
            response_distribution, response_matching_count = calculate_response_distribution(q_id, matching_personas)

            if response_distribution is not None and response_matching_count > 0:
                # Map response codes to response options
                response_percentages = {}
                for code, percentage in response_distribution.items():
                    try:
                        option_text = response_options[int(code)-1]
                    except (IndexError, ValueError):
                        option_text = f"Option {code}"
                    response_percentages[option_text] = percentage

                # Determine the most popular answer
                most_popular_code = response_distribution.idxmax()
                try:
                    most_popular_answer = response_options[int(most_popular_code)-1]
                except (IndexError, ValueError):
                    most_popular_answer = f"Option {most_popular_code}"

                results.append({
                    'Prompt': prompt,
                    'Count from CSV': count,
                    'Matching Personas Count': matching_count,
                    'Question ID': q_id,
                    'Question Text': q_text,
                    'Response Percentages': response_percentages,
                    'Most Popular Answer': most_popular_answer
                })
            else:
                results.append({
                    'Prompt': prompt,
                    'Count from CSV': count,
                    'Matching Personas Count': matching_count,
                    'Question ID': q_id,
                    'Question Text': q_text,
                    'Response Percentages': {},
                    'Most Popular Answer': None
                })

    results_df = pd.DataFrame(results)
    return results_df

# Process all prompts and questions
results_df = process_all_prompts_and_questions(persona_df)

# Display the results
#print(results_df[['Prompt', 'Count from CSV', 'Matching Personas Count', 'Question ID', 'Most Popular Answer']])

# Save the results to a CSV file (optional)
results_df.to_csv('results/persona_survey_results_with_counts.csv', index=False)