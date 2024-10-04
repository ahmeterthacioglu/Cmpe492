import re
import csv
from PyPDF2 import PdfReader

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"Warning: Page in {pdf_path} did not return text.")
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Extract text from Turkish and English questionnaires
turkish_text = extract_text_from_pdf("data/F00009572-WVS7_Questionnaire_Turkey_2018_Turkish.pdf")
english_text = extract_text_from_pdf("data/F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf")

# Function to clean and extract questions, response options, and SHOW CARD sections
def extract_questions(text):
    questions_raw = re.split(r'\nQ(\d+)[\.\s]*', text)
    show_card_pattern = re.compile(r'SHOW CARD.*?(?=\nQ\d+|\Z)', re.DOTALL)
    questions = []
    show_cards = {}

    # Extract SHOW CARD sections
    for match in show_card_pattern.finditer(text):
        show_card_text = match.group(0).strip()
        show_card_id = f"SHOW_CARD_{len(show_cards) + 1}"
        show_cards[show_card_id] = show_card_text

    # Identify the response label header pattern
    response_label_pattern = re.compile(
        r'(?<=\b)(Kesin Kabul Ederim|Kabul Ederim|Kabul Etmem|Kesin Kabul Etmem|Kesinlikle Katılmam|Kesinlikle Kabul Etmem|İstemem|Olabilir|'
        r'Çok Önemli|Biraz Önemli|Pek Önemli Değil|Hiç Önemli Değil|Strongly agree|Agree|Disagree|Strongly disagree|Hard to say|'
        r'Kesinlikle Katılırım|Katılırım|Ne Katılırım Ne Katılmam|Katılmam|'
        r'Aktif üye|Pasif üye|Üye değil|'
        r'İyi Olurdu|Fark Etmezdi|Kötü Olurdu|'
        r'Çok mutluyum|Biraz mutluyum|Pek mutlu değilim|Hiç mutlu değilim|'
        r'Çok iyi|İyi|Fena değil|Kötü|Çok kötü|Biraz kötü|Ne iyi ne kötü|Biraz iyi|Pek sık rastlanmaz|Hiç rastlanmaz|'
        r'Sık Sık|Bazen|Pek Değil|Hiç Değil|'
        r'Tamamen Güvenirim|Biraz Güvenirim|Pek Güvenmem|Hiç Güvenmem|'
        r'Çok az sayıda|Yarıdan az|Çoğunluk|Hemen hemen hepsi|'
        r'Asla|Pek değil|Sıklıkla|Her zaman|'
        r'Çok sık|Epeyce sık|Fazla sık değil|Hiç sık değil|'
        r'Çok Endişe Duyuyorum|Biraz Endişe Duyuyorum|Pek Endişe Duymuyorum|Hiç Endişe Duymuyorum|'
        r'Günlük|Haftalık|Ayda bir|Ayda birden az|Hiç|'
        r'Yaptım Gerekirse|Yapabilirim|Asla Yapmam|'
        r'Genellikle|Hiçbir zaman|Oy kullanma hakkım yok|'
        r'Çok yakın|Yakın|Pek yakın değil|Hiç yakın değil|'
        r'Agree strongly|Neither agree nor disagree|Disagree strongly|'
        r'Good|Don’t mind|Bad|'
        r'Trust completely|Trust somewhat|Do not trust very much|Do not trust at all|'
        r'A great deal|Quite a lot|Not very much|None at all|'
        r'Active member|Inactive member|Don’t belong|'
        r'None of them|Few of them|Most of them|All of them|'
        r'Never|Rarely|Frequently|Always|'
        r'Very good|Quite good|Neither good,nor bad|Quite bad|Very bad|'
        r'Very frequently|Quite frequently|Not frequently|Not at all frequently|'
        r'Very much|A good deal|Not much|Not at all|Very close|Close|Not very close|Not close at all|'
        r'First choice|Second choice|Completely disagree|Completely agree|Fairly good|Fairly bad|'
        r'Daily|Weekly|Monthly|Less than monthly|Have done|Might do|Would never do|Usually|Not allowed to vote|Very often|Fairly often|Not often|Not at all often|'
        r'Mentioned|Not mentioned|Belirtildi|Belirtilmedi|Very important|Rather important|Not very important|Not at all important)'

    )

    current_response_labels = []

    # Loop through questions_raw in steps of 2 to extract question details
    for i in range(1, len(questions_raw), 2):
        segment = questions_raw[i - 1]
        segment = re.sub(r'\s+', ' ', segment).strip()

        if 80 <i< 90:
            print("X---",segment)
        if 80 <i< 90:
            print(questions_raw[i])
        # Look for response labels in the segment before each question

        found_labels = response_label_pattern.findall(segment)
        if found_labels:
            if 80 < i < 90:
                for label in found_labels:
                    print("A-",label)
            # Update current response labels if we find a new set of labels
            current_response_labels = [label.strip() for label in found_labels]

        question_id = f"Q{questions_raw[i].strip()}"
        question_text = questions_raw[i + 1]

        # Clean up excessive whitespace/newlines
        question_text = re.sub(r'\s+', ' ', question_text).strip()
        if 80 <i< 90:
            print("oldques:",question_text)

        # Remove any unwanted segments that should not be part of the question
        exclusion_patterns = [r'(?<=\b)Cevapsız soruların kodları.*$', r'(?<=\b)SHOW CARD.*$']
        for pattern in exclusion_patterns:
            question_text = re.sub(pattern, '', question_text)
        if 80 <i< 90:
            print("newques:",question_text)
        # Extract response values and replace with appropriate labels dynamically
        response_values = re.findall(r'-?\d+', question_text)
        if 80 <i< 90:
            print("responsevalues:",response_values)
            print("current_response_labels",current_response_labels)
        response_options = []
        for value in response_values:
            try:
                if value == "-1":
                    response_options.append("Fikri Yok")
                elif value == "-2":
                    response_options.append("Cevap Yok")
                else:
                    # Use the current response labels to replace numeric values
                    response_label_index = int(value) - 1
                    if -1 <= response_label_index < len(current_response_labels):
                        if 80 < i < 90:
                            print(current_response_labels[response_label_index])
                        response_options.append(current_response_labels[response_label_index])
                    else:
                        response_options.append(value)
            except (IndexError, ValueError):
                response_options.append(value)

        # Associate SHOW CARD if mentioned in the question text
        related_show_card = None
        question_main_cleaned = re.sub(r'(\d+\s+[^\n]+)', '', question_text).strip()
        if 80 <i< 90:
            print("WWWW-",question_main_cleaned)
            print("QQQQ-",response_options)

        for show_card_id, show_card_text in show_cards.items():
            if show_card_id in question_text:
                related_show_card = show_card_text
                break

        # Add the extracted question details to the list
        questions.append((question_id, question_main_cleaned, response_options, related_show_card))

    return questions

# Extract questions from Turkish and English text
turkish_questions = extract_questions(turkish_text)
english_questions = extract_questions(english_text)

# Function to map Turkish questions to English questions
def map_questions(turkish_questions, english_questions):
    mapping = []
    min_len = min(len(turkish_questions), len(english_questions))

    for i in range(min_len):
        turkish_qid, turkish_text, turkish_options, turkish_show_card = turkish_questions[i]
        english_qid, english_text, english_options, english_show_card = english_questions[i]

        # Map questions assuming correct ordering
        mapping.append({
            "Turkish Question ID": turkish_qid,
            "Turkish Question Text": turkish_text,
            "Turkish Response Options": turkish_options,
            "Turkish Show Card": turkish_show_card,
            "English Question ID": english_qid,
            "English Question Text": english_text,
            "English Response Options": english_options,
            "English Show Card": english_show_card
        })

    return mapping

# Create a mapping between Turkish and English questions
question_mapping = map_questions(turkish_questions, english_questions)

# Save questions into a CSV for further processing and manual checking
def save_questions_to_csv(mapping, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Turkish Question ID",
            "Turkish Question Text",
            "Turkish Response Options",
            "Turkish Show Card",
            "English Question ID",
            "English Question Text",
            "English Response Options",
            "English Show Card"
        ])
        writer.writeheader()
        for row in mapping:
            writer.writerow(row)

# Save the mapped questions to a CSV file
save_questions_to_csv(question_mapping, "results/survey_question_mapping.csv")

# Identify inconsistencies and manually verify them
def identify_inconsistencies(mapping):
    inconsistencies = []

    for item in mapping:
        # Check if both Turkish and English questions have similar response options count
        if len(item["Turkish Response Options"]) != len(item["English Response Options"]):
            inconsistencies.append(item)

        # Check if there is a SHOW CARD mismatch
        if (item["Turkish Show Card"] and not item["English Show Card"]) or (
                item["English Show Card"] and not item["Turkish Show Card"]):
            inconsistencies.append(item)

    return inconsistencies

# Get inconsistencies
inconsistencies = identify_inconsistencies(question_mapping)

# Save inconsistencies to a separate CSV file for manual review
def save_inconsistencies_to_csv(inconsistencies, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Turkish Question ID",
            "Turkish Question Text",
            "Turkish Response Options",
            "Turkish Show Card",
            "English Question ID",
            "English Question Text",
            "English Response Options",
            "English Show Card"
        ])
        writer.writeheader()
        for row in inconsistencies:
            writer.writerow(row)

save_inconsistencies_to_csv(inconsistencies, "results/inconsistencies_review.csv")

print("Extraction and mapping completed. Please check 'survey_question_mapping.csv' and 'inconsistencies_review.csv' for details.")

# Main script entry point
if __name__ == '__main__':
    print("Running the survey extraction script...")