import pandas as pd
from itertools import product
import random

# Step 1: Load the survey data file
survey_data = pd.read_csv('../data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')

# Step 2: Define the mapping for the features with expanded options

# Gender mapping
gender_map = {1: 'Erkek', 2: 'Kadın'}

# Age groups
def categorize_age(age):
    if age <= 29:
        return '30 yaşından küçük'
    elif 30 <= age <= 49:
        return '30-50 yaş arası'
    else:
        return '50 yaşında veya daha yaşlı'

# Expanded Marital status mapping
marital_status_map = {
    1: 'Evli',
    2: 'Evli gibi birlikte yaşamakta',
    3: 'Boşanmış',
    4: 'Evli fakat eşinden ayrı yaşıyor',
    5: 'Eşi ölmüş yani dul',
    6: 'Bekâr'
}

# Expanded Children count mapping
def categorize_children(children):
    if pd.isna(children):
        return None
    elif children >= 1:
        return 'Çocuk sahibi'
    else:
        return 'Çocuğu yok'

# Expanded Education levels mapping
education_level_map = {
    792001: 'Hiç okula gitmemiş',
    792014: 'İlkokuldan ayrılmış',
    792015: 'İlkokul mezunu',
    792016: 'Ortaokuldan ayrılmış',
    792017: 'Ortaokul mezunu',
    792018: 'Liseden ayrılmış',
    792019: 'Lise mezunu',
    792020: 'Üniversiteden ayrılmış',
    792021: 'Üniversite mezunu',
    792022: 'Lisansüstü (master veya doktora derecesi var)',
    792023: 'Şu anda lise öğrencisi',
    792024: 'Şu anda üniversite öğrencisi'
}

# Expanded Employment status mapping
employment_status_map = {
    1: 'Ücretli ve tam zamanlı (yani haftada 30 saatten fazla) çalışıyor',
    2: 'Ücretli ve yarı zamanlı (yani haftada 30 saatten az) çalışıyor',
    3: 'Kendi işinin sahibi',
    4: 'Emekli',
    5: 'Ev kadını',
    6: 'Öğrenci',
    7: 'İşsiz/iş arıyor',
    8: 'Diğer (...)'
}

# Expanded Social class mapping
social_class_map = {
    1: 'Üst sınıf',
    2: 'Orta sınıfın üst kısmında',
    3: 'Orta sınıfın alt kısmında',
    4: 'Çalışan, işçi, emekçi sınıfı',
    5: 'Alt sınıf'
}

# Settlement size mapping (5 groups)
settlement_size_map = {
    1: '5,000’den daha az kişi',
    2: '5,000 - 19,999 kişi arası',
    3: '20,000 - 99,999 kişi arası',
    4: '100,000 - 499,999 kişi arası',
    5: '500,000 veya daha fazla kişi'
}

# Step 3: Apply the mapping to the survey data to create the feature columns
survey_data['Cinsiyet'] = survey_data['Q260'].map(gender_map)
survey_data['Yaş Grubu'] = survey_data['Q262'].apply(categorize_age)
survey_data['Medeni Durum'] = survey_data['Q273'].map(marital_status_map)
survey_data['Çocuk Sahipliği'] = survey_data['Q274'].apply(categorize_children)
survey_data['Eğitim Düzeyi'] = survey_data['Q275A'].map(education_level_map)
survey_data['İş Durumu'] = survey_data['Q279'].map(employment_status_map)
survey_data['Sosyal Sınıf'] = survey_data['Q287'].map(social_class_map)
survey_data['Yerleşim Yeri'] = survey_data['G_TOWNSIZE2'].map(settlement_size_map)

# Step 4: Filter out rows with any missing demographic data once
survey_data_filtered = survey_data.dropna(subset=[
    'Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
    'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri'
])

# Step 5: Group the filtered data by the features for faster persona matching
grouped_data = survey_data_filtered.groupby([
    'Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
    'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri'
]).size().reset_index(name='Count')

# Step 6: Define all possible combinations of personas and add prompts
persona_counts = []
total_personas = 0

def create_prompt(row):
    # Define a random name for example purposes
    name = "Ayşe" if row['Cinsiyet'] == "Kadın" else "Ahmet"
    
    # Generate random age based on the age group
    if row['Yaş Grubu'] == '30 yaşından küçük':
        age = random.randint(18, 29)
    elif row['Yaş Grubu'] == '30-50 yaş arası':
        age = random.randint(30, 49)
    else:
        age = random.randint(50, 90)
    
    # Generate random settlement size based on the category, with only the two most significant digits random
    if row['Yerleşim Yeri'] == '5,000’den daha az kişi':
        population = random.randint(5, 49) * 100  # 500 to 4900
    elif row['Yerleşim Yeri'] == '5,000 - 19,999 kişi arası':
        population = random.randint(6, 19) * 1000  # 5000 to 19000
    elif row['Yerleşim Yeri'] == '20,000 - 99,999 kişi arası':
        population = random.randint(21, 99) * 1000  # 20000 to 99000
    elif row['Yerleşim Yeri'] == '100,000 - 499,999 kişi arası':
        population = random.randint(11, 49) * 10000  # 100000 to 490000
    else:
        population = random.randint(6, 50) * 100000  # 500000 to 5000000

    
    # Create the personalized prompt
    prompt = (
        f"{name}, {age} yaşında {row['Medeni Durum'].lower()} bir {row['Cinsiyet'].lower()}, "
        f"{row['Çocuk Sahipliği'].lower()}, {row['Eğitim Düzeyi'].lower()}, "
        f"{population} nüfusa sahip bir şehirde yaşıyor ve kendisini {row['Sosyal Sınıf'].lower()} olarak tanımlıyor."
    )
    
    return prompt

for persona in product(
        ['Erkek', 'Kadın'],
        ['30 yaşından küçük', '30-50 yaş arası', '50 yaşında veya daha yaşlı'],
        list(marital_status_map.values()),
        ['Çocuğu yok', 'Çocuk sahibi'],
        list(education_level_map.values()),
        list(employment_status_map.values()),
        list(social_class_map.values()),
        list(settlement_size_map.values())
):
    (gender, age_group, marital_status, children_group, education_level,
     employment_status, social_class, settlement_size) = persona
    
    match = grouped_data[
        (grouped_data['Cinsiyet'] == gender) &
        (grouped_data['Yaş Grubu'] == age_group) &
        (grouped_data['Medeni Durum'] == marital_status) &
        (grouped_data['Çocuk Sahipliği'] == children_group) &
        (grouped_data['Eğitim Düzeyi'] == education_level) &
        (grouped_data['İş Durumu'] == employment_status) &
        (grouped_data['Sosyal Sınıf'] == social_class) &
        (grouped_data['Yerleşim Yeri'] == settlement_size)
    ]
    
    count = match['Count'].values[0] if not match.empty else 0
    if count > 0:
        total_personas += count
        persona_data = {
            'Cinsiyet': gender,
            'Yaş Grubu': age_group,
            'Medeni Durum': marital_status,
            'Çocuk Sahipliği': children_group,
            'Eğitim Düzeyi': education_level,
            'İş Durumu': employment_status,
            'Sosyal Sınıf': social_class,
            'Yerleşim Yeri': settlement_size,
            'Count': count
        }
        
        persona_data['Prompt'] = create_prompt(persona_data)
        persona_counts.append(persona_data)

persona_df = pd.DataFrame(persona_counts)

# Step 8: Display the total matched personas
print(f"Total matched personas: {total_personas}")

# Save the DataFrame to CSV
persona_df.to_csv('../results/important_persona_counts_with_prompts.csv', index=False)
