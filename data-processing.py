from datasets import load_dataset
import pandas as pd
import re
import pickle
from haystack.dataclasses import Document

ds = load_dataset("kayrab/patient-doctor-qa-tr-167732", split="train", use_auth_token=None)
print(ds)
print(ds.column_names)
print(ds[0])

df = ds.to_pandas()

# Boş veya eksik değerleri sil
df = df.dropna(subset=['question_content', 'question_answer'])
df = df[df['question_content'].str.strip() != '']
df = df[df['question_answer'].str.strip() != '']

print("Temizlenmiş veri boyutu:", df.shape)

df = df.drop_duplicates(subset=['question_content', 'question_answer'])

print("Tekrarlar silindikten sonraki veri boyutu:", df.shape)

def clean_text(text):

    # NaN veya olmayan değerleri string'e dönüştür
    if not isinstance(text, str):
        text = str(text)

    text = re.sub(r'\s+', ' ', text)  # Birden fazla boşluk, sekme veya \n karakterini tek boşluğa indirir
    text = text.strip()  # Metnin başındaki ve sonundaki tüm boşlukları siler
    return text

df['question_content'] = df['question_content'].apply(clean_text)
df['question_answer'] = df['question_answer'].apply(clean_text)

print("Metin temizliği tamamlandı.")

df.to_csv("cleaned_patient_doctor_qa.csv", index=False)

def create_documents_and_save(csv_path="cleaned_patient_doctor_qa.csv"):
    """
    Temizlenmiş CSV dosyasını okur, Haystack Document nesnelerine dönüştürür
    ve bu listeyi ileride hızlı yükleme için diske (documents.pkl) kaydeder.
    """

    try:
        # CSV dosyasını okuma
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"HATA: '{csv_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return []

    documents = []
    for _, row in df.iterrows():
        # Doctor cevabını (question_answer) RAG Context'i olarak kullanma
        content = str(row['question_answer'])

        # Meta verisi ekleyerek arama kabiliyetlerini artırma
        meta = {
            "question": str(row['question_content']),
            "doctor_title": str(row['doctor_title']),
            "doctor_speciality": str(row['doctor_speciality']),
        }
        documents.append(Document(content=content, meta=meta))

    print(f"{len(documents)} benzersiz belge oluşturuldu.")

    # İleri aşamada (Vektör Veritabanı oluştururken) hızlı yükleme için listeyi kaydetme
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("Belge listesi başarıyla kaydedildi: documents.pkl")

    return documents

if __name__ == "__main__":
    documents_list = create_documents_and_save()

