import nest_asyncio
nest_asyncio.apply()
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
import streamlit as st

from rag_index_builder import load_existing_store, EMBEDDING_MODEL
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.components.common.google_genai.utils import Secret

# Ortam değişkenlerini yükle
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_DIM = 384


# Rag pipeline kurulumu
@st.cache_resource
def initialize_rag_pipeline():
    """
    Qdrant DocumentStore'u yükler ve RAG Sorgulama Pipeline'ını kurar.
    Bu fonksiyon Streamlit cache'i sayesinde sadece bir kez çalışır!
    """

    if not GOOGLE_API_KEY:
        st.error("HATA: GOOGLE_API_KEY yüklenemedi. Lütfen .env dosyanızı kontrol edin.")
        return None

    # 1. Qdrant DocumentStore'u Yükle
    with st.spinner("Qdrant Veritabanı hızlıca yükleniyor..."):
        try:
            document_store = load_existing_store()

            if document_store is None:
                st.warning(
                    "Veritabanı yüklenemedi. Lütfen 'rag_index_builder.py' dosyasını çalıştırdığınızdan emin olun.")
                return None

        except RuntimeError as e:
            if "already accessed" in str(e):
                st.error("Qdrant veritabanı zaten başka bir süreç tarafından kullanılıyor.")
                st.info("Çözüm: Tüm Python süreçlerini kapatıp uygulamayı yeniden başlatın.")
                st.stop()
            else:
                st.error(f"Beklenmeyen hata: {e}")
                return None

    # Sorgulama Pipeline'ını kurma

    # Retriever, Qdrant'tan arama yapacak bileşen
    retriever = QdrantEmbeddingRetriever(document_store=document_store, top_k=3)
    # Sorgu metnini vektörlere çevirecek bileşen
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)

    try:
        gemini_secret = Secret.from_env_var("GOOGLE_API_KEY")
    except ValueError:
        st.error("HATA: GOOGLE_API_KEY ortam değişkeni ayarlanmadı veya .env dosyası okunmadı.")
        return None

    # Generator (LLM)
    generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash-exp", api_key=gemini_secret)
    prompt_template = [
        ChatMessage.from_system(
            "Sen bir Türkçe sağlık danışmanlığı chatbotusun. "
            "Rolün, yalnızca sağlanan doktor yanıtlarını analiz ederek hastanın sorusuna uygun açıklama üretmektir. "
            "Kendi tıbbi yorumunu ekleme, varsayım yapma veya belgelerde yer almayan bilgileri kullanma. "
            "Her yanıtın anlaşılır, sakin ve profesyonel bir tonda olmalı. "
            "Eğer belgelerde yeterli veya ilgili bilgi yoksa, bunu açıkça belirt ('Sağlanan bilgilere göre bu konuda yeterli veri bulunmamaktadır. Lütfen doktorunuza danışınız.') "
            "Başka bir şey yazma. "
            "Eğer belgelerde yeterli bilgi varsa, yanıtını yapılandırılmış biçimde ve empatik bir kapanışla ver."
        ),
        ChatMessage.from_user(
            """Belgeler:
    {% for doc in documents %}
    - Uzmanlık Alanı: {{ doc.meta['doctor_speciality'] }}
    - Orijinal Soru: {{ doc.meta['question'] }}
    - Cevap İçeriği: {{ doc.content }}
    {% endfor %}

    Soru: {{question}}

    Yanıt formatı:

    - Eğer belgelerde yeterli bilgi **yoksa**:
      "Sağlanan bilgilere göre bu konuda yeterli veri bulunmamaktadır. Lütfen doktorunuza danışınız." ifadesini **aynen** yaz. Başka hiçbir şey ekleme.

    - Eğer belgelerde yeterli bilgi **varsa**:
      Sağlanan bilgilere göre:
      1. **Kısa Özet:** (sorunun genel yanıtını 1-2 cümlede açıkla)
      2. **Detaylı Açıklama:** (doktor cevaplarına dayalı detayları ve olası nedenleri açıkla)
      3. **Uyarı veya Öneri:** (doktorların vurguladığı önemli noktaları belirt)
      4. **Kaynakça:** 
         - Yalnızca yanıtla doğrudan ilişkili uzmanlık alanlarını listele.
      5. **Empatik Kapanış:** "Geçmiş olsun." ifadesini ekle.
    """
        )
    ]

    prompt_builder = ChatPromptBuilder(template=prompt_template)

    # RAG Sorgu Pipeline'ı Oluşturma ve Bağlama
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("generator", generator)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "generator.messages")

    st.sidebar.success("RAG Sistemi Hazır!")
    return rag_pipeline


# Streamlit uygulaması:
def main():
    st.set_page_config(page_title="🩺 Türkçe Hasta-Doktor Chatbotu", page_icon="🇹🇷", layout="wide")

    # Sidebar: Proje Bilgileri ve Status
    st.sidebar.title("📊 Proje Durumu")
    st.sidebar.markdown(f"**Veritabanı:** Qdrant (Kalıcı)")
    st.sidebar.markdown(f"**Vektör Boyutu:** 768")
    st.sidebar.markdown(f"**Belge Sayısı:** 332.036")

    # Ana Başlık
    st.markdown("## Türkçe Hasta-Doktor Sağlık Asistanı")
    st.caption(
        "Bu chatbot, doktorların verdiği kanıtlara dayalı olarak yanıt üretir. Lütfen tıbbi tavsiye almadığınızı unutmayın.")

    # RAG Pipeline'ı Yükle/Oluştur
    rag_pipeline = initialize_rag_pipeline()

    if rag_pipeline is None:
        return

    # Sohbet geçmişi (Streamlit session state)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if prompt := st.chat_input("Örn: Baş ağrısı için ne zaman doktora gitmeliyim?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG boru hattını çalıştır ve yanıt al
        with st.chat_message("assistant"):
            with st.spinner("İlgili doktor cevapları taranıyor ve yanıt oluşturuluyor..."):
                try:
                    result = rag_pipeline.run({
                        "text_embedder": {"text": prompt},
                        "prompt_builder": {"question": prompt}
                    })

                    response = result["generator"]["replies"][0].text
                    st.markdown(response)

                except Exception as e:
                    st.error(f"Sorgu işlenirken bir hata oluştu: {str(e)}")
                    response = "Sorgu hatası."

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
