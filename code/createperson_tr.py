import pandas as pd
import random

survey_data = pd.read_csv('data/F00013167-WVS_Wave_7_Turkey_Csv_v5.0.csv', sep=';')

male_names = ["Mehmet", "Mustafa", "Ahmet", "Ali", "Hüseyin", "Hasan", "İbrahim", "İsmail", "Osman", "Yusuf",
                 "Murat", "Ömer", "Ramazan", "Halil", "Süleyman", "Abdullah", "Mahmut", "Recep", "Salih", "Fatih",
                 "Kadir", "Emre", "Hakan", "Adem", "Kemal", "Yaşar", "Bekir", "Musa", "Metin", "Serkan"]
female_names = ["Fatma", "Ayşe", "Emine", "Hatice", "Zeynep", "Elif", "Meryem", "Selma", "Şerife", "Zehra", "Sultan",
                 "Hanife", "Merve", "Havva", "Zeliha", "Esra", "Fadime", "Özlem", "Hacer", "Yasemin", "Melek", "Rabia",
                 "Hülya", "Cemile", "Sevim", "Gülsüm", "Leyla", "Dilek", "Büşra", "Aysel"]

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

# Marital status mapping
def marital_status_map(index):
    if index in [1, 2]:
        return 'Evli'
    elif index in [3, 4, 5, 6]:
        return 'Bekâr'
    elif index == -1:
        return '' #Fikri Yok
    elif index == -2:
        return '' #Cevap Yok  
    return None

# Children count mapping
def categorize_children(children):
    if pd.isna(children):
        return None
    elif children >= 1:
        return 'Çocuk sahibi'
    else:
        return 'Çocuğu yok'

# Education levels mapping
def education_level_map(index):
    if index == 792001:
        return 'Hiç okula gitmemiş'
    elif index == 792014:
        return 'Ilkokul terk'
    elif index in [792015, 792016]:
        return 'Ilkokul mezunu'
    elif index in [792017, 792018]:
        return 'Ortaokul mezunu'
    elif index in [792019, 792020]:
        return 'Lise mezunu'
    elif index == 792021:
        return 'Universite mezunu'
    elif index in [792023, 792024]:
        return 'Şu anda öğrenci'
    elif index == -1:
        return '' #Fikri Yok
    elif index == -2:
        return '' #Cevap Yok  
    return None

# Employment status mapping
employment_status_map = {
    1: 'Ucretli ve tam zamanlı çalışan',
    2: 'Ucretli ve yarı zamanlı çalışan',
    3: 'Kendi işinin sahibi',
    4: 'Emekli',
    5: 'Ev kadını',
    6: 'Oğrenci',
    7: 'Işsiz/iş arayan',
    8: 'Diğer',
    -1: '', #Fikri Yok
    -2: '' #Cevap Yok
}

# Social class mapping
social_class_map = {
    1: 'Ust sınıf',
    2: 'Orta sınıfın üst kısmında',
    3: 'Orta sınıfın alt kısmında',
    4: 'Çalışan, işçi, emekçi sınıfı',
    5: 'Alt sınıf',
    -1: '', #Fikri Yok
    -2: '' #Cevap Yok
}

# Settlement type mapping
settlement_type_map = {
    1: 'Kent merkezinde',
    2: 'Kırsal alanda',
    -1: '', #Fikri Yok
    -2: '' #Cevap Yok
}

# Region mapping from N_REGION_WVS
region_map = {
    792101: "TR10: İstanbul",
    792102: "TR21: Tekirdağ, Edirne, Kırklareli",
    792103: "TR22: Balıkesir, Çanakkale",
    792104: "TR31: İzmir",
    792105: "TR32: Aydın, Denizli, Muğla",
    792106: "TR33: Manisa, Afyon, Kütahya, Uşak",
    792107: "TR41: Bursa, Eskişehir, Bilecik",
    792108: "TR42: Kocaeli, Sakarya, Düzce, Bolu, Yalova",
    792109: "TR51: Ankara",
    792110: "TR52: Konya, Karaman",
    792111: "TR61: Antalya, Isparta, Burdur",
    792112: "TR62: Adana, Mersin",
    792113: "TR63: Hatay, Kahramanmaraş, Osmaniye",
    792114: "TR71: Kırıkkale, Aksaray, Niğde, Kırşehir, Nevşehir",
    792115: "TR72: Kayseri, Sivas, Yozgat",
    792116: "TR81: Zonguldak, Karabük, Bartın",
    792117: "TR82: Kastamonu, Çankırı, Sinop",
    792118: "TR83: Samsun, Tokat, Çorum, Amasya",
    792119: "TR90: Trabzon, Ordu, Giresun, Rize, Artvin, Gümüşhane",
    792120: "TRA1: Erzurum, Erzincan, Bayburt",
    792121: "TRA2: Ağrı, Kars, Iğdır, Ardahan",
    792122: "TRB1: Malatya, Elazığ, Bingöl, Tunceli",
    792123: "TRB2: Van, Muş, Bitlis, Hakkari",
    792124: "TRC1: Gaziantep, Adıyaman, Kilis",
    792125: "TRC2: Şanlıurfa, Diyarbakır",
    792126: "TRC3: Mardin, Batman, Şırnak, Siirt"
}

survey_data['Cinsiyet'] = survey_data['Q260'].map(gender_map)
survey_data['Yaş Grubu'] = survey_data['Q262'].apply(categorize_age)
survey_data['Medeni Durum'] = survey_data['Q273'].apply(marital_status_map)
survey_data['Çocuk Sahipliği'] = survey_data['Q274'].apply(categorize_children)
survey_data['Eğitim Düzeyi'] = survey_data['Q275A'].apply(education_level_map)
survey_data['İş Durumu'] = survey_data['Q279'].map(employment_status_map)
survey_data['Sosyal Sınıf'] = survey_data['Q287'].map(social_class_map)
survey_data['Yerleşim Yeri'] = survey_data['H_URBRURAL'].map(settlement_type_map)
survey_data['Bölge'] = survey_data['N_REGION_WVS'].map(region_map)

survey_data_filtered = survey_data.dropna(subset=[
    'Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
    'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri', 'Bölge'
])

grouped_data = survey_data_filtered.groupby([
    'Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Çocuk Sahipliği',
    'Eğitim Düzeyi', 'İş Durumu', 'Sosyal Sınıf', 'Yerleşim Yeri', 'Bölge'
]).size().reset_index(name='Count')

def create_prompt(row):
    name = random.choice(female_names) if row['Cinsiyet'] == "Kadın" else random.choice(male_names)
    if row['Yaş Grubu'] == '30 yaşından küçük':
        age = random.randint(18, 29)
    elif row['Yaş Grubu'] == '30-50 yaş arası':
        age = random.randint(30, 49)
    else:
        age = random.randint(50, 90)
    
    # Extract city names from region
    region_info = row['Bölge'].split(': ')[1]
    cities = region_info.split(', ')
    city = random.choice(cities)
    
    prompt = (
        f"{name}, {age} yaşında {row['Medeni Durum'].lower()} bir {row['Cinsiyet'].lower()}, "
        f"{row['Çocuk Sahipliği'].lower()}, {row['Eğitim Düzeyi'].lower()}, "
        f"{city} şehrinde {row['Yerleşim Yeri'].lower()} yaşayan, "
        f"kendi sosyal sınıfını {row['Sosyal Sınıf'].lower()} olarak tanımlayan, {row['İş Durumu'].lower()} birisidir."
    )
    return prompt

grouped_data['Prompt'] = grouped_data.apply(create_prompt, axis=1)

total_personas = grouped_data['Count'].sum()
print(f"Total matched personas: {total_personas}")

grouped_data.to_csv('results/persona_counts_with_prompts_tr.csv', index=False)