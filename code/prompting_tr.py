import pandas as pd
import numpy as np
import ast
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

# Load the processed persona data and survey data
persona_df = pd.read_csv('results/persona_counts_with_prompts_tr.csv')
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
    text = unicodedata.normalize('NFD', text)
    # Replace special Turkish characters
    for src, target in replacements.items():
        text = text.replace(src, target)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.casefold()
    return text

def strip_accents(text):
    text = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in text if not unicodedata.combining(c)])

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

survey_data['Bölge'] = survey_data['Bölge'].astype(str)
survey_data['Bölge'] = survey_data['Bölge'].fillna('')
persona_df['Bölge'] = persona_df['Bölge'].str.replace(r'^TR\d+: ', '', regex=True).str.strip()

persona_df['İş Durumu'] = persona_df['İş Durumu'].str.replace(r'\s*\(.*\)', '', regex=True).str.strip()

string_columns = ['Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
                  'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri', 'Bölge']

for col in string_columns:
    persona_df[col] = persona_df[col].str.strip()

def parse_input(prompt):
    prompt = re.sub(r'\(.*?\)', '', prompt)
    
    prompt_norm = normalize_text(prompt)
    
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
    
    matching_personas['Bölge'] = matching_personas['Bölge'].astype(str).fillna('')
    
    for feature, value in features.items():
        if value is not None and feature in matching_personas.columns:
            if feature == 'Bölge':
                matching_personas = matching_personas[matching_personas['Bölge'].str.contains(value, na=False)]
            else:
                matching_personas = matching_personas[matching_personas[feature] == value]
    return matching_personas

def calculate_response_distribution(question_id, matching_personas):
    if matching_personas is None:
        filtered_data = survey_data.copy()
        # Remove rows with missing responses to the question
        filtered_data = filtered_data[filtered_data[question_id].notna()]
    else:
        # Remove duplicates in matching_personas
        matching_personas_unique = matching_personas[[
            'Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
            'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri', 'Bölge'
        ]].drop_duplicates()

        # Strip whitespace and ensure consistent data types
        for col in matching_personas_unique.columns:
            if matching_personas_unique[col].dtype == 'object':
                matching_personas_unique[col] = matching_personas_unique[col].str.strip().astype(str)
                survey_data[col] = survey_data[col].astype(str).str.strip()

        # Drop rows with missing values in the features being matched
        features_to_check = matching_personas_unique.columns.tolist()
        survey_data_filtered = survey_data.dropna(subset=features_to_check)

        # Merge matching personas with survey data
        filtered_data = survey_data_filtered.merge(
            matching_personas_unique,
            on=features_to_check,
            how='inner'
        ).drop_duplicates()

    if not filtered_data.empty and question_id in filtered_data.columns:
        filtered_data = filtered_data[filtered_data[question_id].notna()]
        response_counts = filtered_data[question_id].value_counts(normalize=True) * 100
        response_counts = response_counts.sort_index()
        matching_count = len(filtered_data)
        return response_counts, matching_count
    else:
        return pd.Series(dtype=float), 0

survey_questions = [q for q in survey_mapping['Turkish Question Text'].tolist() if pd.notna(q)]

# Vectorize the survey questions
vectorizer = TfidfVectorizer().fit(survey_questions)
survey_vectors = vectorizer.transform(survey_questions)

def find_question_id_by_text(question_text, similarity_threshold=0.75):
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
        question_id = survey_mapping.iloc[max_similarity_index]['Turkish Question ID']
        print(f"Matched question with similarity score: {max_similarity_score}")
        return question_id
    else:
        print("No matching question found above the similarity threshold.")
        return None

def get_question_text_and_options(question_id):
    # Find the question in the survey mapping
    question_row = survey_mapping[survey_mapping['Turkish Question ID'] == question_id]
    if not question_row.empty:
        question_text = question_row.iloc[0]['Turkish Question Text']
        response_options = ast.literal_eval(question_row.iloc[0]['Turkish Response Options'])
        return question_text, response_options
    else:
        return None, None

def get_survey_results_with_text(prompt, question_text):
    # Find the question ID based on the provided question text
    question_id = find_question_id_by_text(question_text)
    if not question_id:
        print("Question not found.")
        return None
    prompt = re.sub(r'\(.*?\)', '', prompt)
    features = parse_input(prompt)
    print(f"Extracted features: {features}")

    # Check if all features are None
    if all(value is None for value in features.values()):
        print("No features extracted from prompt. Using all survey respondents.")
        matching_personas = None
    else:
        # Match personas based on the extracted features
        matching_personas = match_personas(features)
        if matching_personas.empty:
            print("No matching personas found.")
            return None
        else:
            print(f"Number of matching personas: {len(matching_personas)}")

    # Calculate the survey response distribution for the given question
    response_distribution, matching_count = calculate_response_distribution(question_id, matching_personas)
    if response_distribution is not None and matching_count > 0:
        # Get the question text and options
        question_text, response_options = get_question_text_and_options(question_id)
        if question_text and response_options:
            print("Response options:")
            for idx, option in enumerate(response_options, start=1):
                print(f"{idx}. {option}")

        print(f"\nSurvey results for question {question_id} based on the matching personas:")
        print(f"Number of persons matching the criteria: {matching_count}")
        print(response_distribution)
        return response_distribution
    else:
        print(f"No matching data found for question {question_id}.")
        return None

# Example usage
question_text = "Şimdi soracaklarımdan her biri sizin için ne kadar önemlidir? Yani bunlar sizin için çok mu önemli, biraz mı önemli, pek önemli değil mi, yoksa hiç önemi yok mu? Aileniz"

prompt = "Ayşe, 49 yaşında evli bir kadın, çocuk sahibi, ilkokul mezunu, İstanbul şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını orta sınıfın alt kısmında olarak tanımlayan, ev kadını birisidir."
prompt1="Ayşe, 49 yaşında evli bir kadın"
prompt2="Ayşe, 45 yaşında evli bir kadın, çocuk sahibi, ilkokul mezunu, İstanbul şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını orta sınıfın üst kısmında olarak tanımlayan, ev kadını birisidir."
prompt3="Ayşe, 47 yaşında evli bir kadın, çocuk sahibi, ilkokul mezunu, İstanbul şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını çalışan, işçi, emekçi sınıfı olarak tanımlayan ev kadını birisidir."
prompt4="Ayşe, 32 yaşında evli bir kadın, çocuk sahibi, üniversite mezunu, İstanbul şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını orta sınıfın üst kısmında olarak tanımlayan ücretli ve tam zamanlı çalışan birisidir."
prompt5="Ayşe, 84 yaşında evli bir kadın, çocuk sahibi, hiç okula gitmemiş, İstanbul şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını çalışan, işçi, emekçi sınıfı olarak tanımlayan ev kadını birisidir."
prompt6="Ahmet, 18 yaşında bekâr bir erkek, çocuğu yok, lise mezunu, İzmir şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını orta sınıfın alt kısmında olarak tanımlayan ücretli ve tam zamanlı çalışan birisidir."
prompt7="Ahmet, 22 yaşında bekâr bir erkek, çocuğu yok, universite mezunu, Muğla şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını orta sınıfın alt kısmında olarak tanımlayan, ucretli ve tam zamanlı çalışan birisidir."
prompt8="Ayşe, 31 yaşında evli bir kadın, çocuk sahibi, ilkokul mezunu, Kahramanmaraş şehrinde kent merkezinde yaşayan, kendi sosyal sınıfını çalışan, işçi, emekçi sınıfı olarak tanımlayan, ev kadını birisidir."

prompts = [prompt,prompt1, prompt2, prompt3, prompt4, prompt5, prompt6,prompt7,prompt8]
for prompt in prompts:
    results_text = get_survey_results_with_text(prompt, question_text)
    if results_text is not None:
        print(results_text)
