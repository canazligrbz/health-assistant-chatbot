import os
import time
import pickle
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from tqdm import tqdm

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "turkish_health_qa"
PICKLE_PATH = "documents.pkl" # data-processing.py'nin çıktısı
EMBEDDING_DIM = 384 # paraphrase-multilingual-MiniLM-L12-v2 modelinin boyutu



def load_documents_from_pickle(pickle_path=PICKLE_PATH) -> list[Document]:
    """
    Kaydedilen Document listesini diskten hızlıca yükler.
    """

    if not os.path.exists(pickle_path):
        print(f"HATA: {pickle_path} dosyası bulunamadı. Lütfen önce setup_vector_store.py dosyasını çalıştırın.")
        return []

    try:
        with open(pickle_path, "rb") as f:
            documents = pickle.load(f)
        print(f"[{len(documents)}] benzersiz belge (Document) yüklendi.")
        return documents
    except Exception as e:
        print(f"Belgeler yüklenirken hata: {e}")
        return []


def create_and_index_documents(documents: list[Document], batch_size=1000):
    """
    documents.pkl'den yüklenen Haystack Document'lerini batch'ler halinde Qdrant'a yazar.
    """

    start_time = time.time()

    print("=" * 80)
    print("TÜRKÇE SAĞLIK QA INDEXLEME SÜRECİ")
    print("=" * 80)

    total_docs = len(documents)

    # 2. Qdrant DocumentStore oluştur
    print(f"\n2. Qdrant veritabanı oluşturuluyor: {QDRANT_PATH}")

    # Eğer varsa sil ve yeniden oluştur
    if os.path.exists(QDRANT_PATH):
        print("Mevcut veritabanı siliniyor...")
        import shutil
        shutil.rmtree(QDRANT_PATH)

    document_store = QdrantDocumentStore(
        path=QDRANT_PATH,
        index=COLLECTION_NAME,
        embedding_dim=384,  # paraphrase-multilingual-MiniLM-L12-v2 boyutu
        recreate_index=True,
        return_embedding=True,
        wait_result_from_api=True
    )
    print("Veritabanı oluşturuldu")

    # 3. Embedder'ı hazırla
    print(f"\n3. Embedding modeli yükleniyor: {EMBEDDING_MODEL}")
    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL,
        progress_bar=False,
        batch_size=32
    )
    embedder.warm_up()
    print("Model hazır")

    # 4. Belgeleri batch'ler halinde işle
    print(f"\n4. Belgeler işleniyor ve indexleniyor (batch_size={batch_size})...")

    total_indexed = 0

    for start_idx in tqdm(range(0, total_docs, batch_size), desc="Batch'ler işleniyor"):
        end_idx = min(start_idx + batch_size, total_docs)

        # Python listesini dilimleme (slicing) ile alıyoruz
        batch_documents = documents[start_idx:end_idx]

        # Embedding oluştur
        # Artık batch_documents'ı kullanıyoruz
        docs_with_embeddings = embedder.run(documents=batch_documents)

        # Qdrant'a yaz
        document_store.write_documents(docs_with_embeddings["documents"])

        total_indexed += len(batch_documents)

    print(f"\nİndexleme tamamlandı!")
    print(f"Toplam indexlenen belge: {total_indexed}")
    print(f"Veritabanı konumu: {os.path.abspath(QDRANT_PATH)}")

    # 5. Doğrulama
    print("\n5. Veritabanı doğrulaması...")
    final_count = document_store.count_documents()
    print(f"Veritabanındaki belge sayısı: {final_count}")

    # İlk belgeyi göster
    sample_docs = list(document_store._client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1
    )[0])

    print("\n" + "=" * 80)
    print("BAŞARILI! Artık app.py'yi çalıştırabilirsiniz.")
    print("=" * 80)

def load_existing_store():
    """
    Mevcut Qdrant veritabanını hızlıca yükler ve döndürür.
    """

    if not os.path.exists(QDRANT_PATH):
        print("Qdrant veritabanı bulunamadı. Lütfen önce indexleme işlemini tamamlayın.")
        return None

    try:
        document_store = QdrantDocumentStore(
            path=QDRANT_PATH,
            index=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM,
            return_embedding=True
        )
        print("Qdrant veritabanı başarıyla yüklendi.")
        return document_store
    except Exception as e:
        print(f"Veritabanı yüklenirken hata oluştu: {e}")
        return None


if __name__ == "__main__":
    # 1. Belgeleri yükle
    documents_list = load_documents_from_pickle()

    # 2. Qdrant Store'u oluştur ve indeksle
    if documents_list:
        # Tüm belgeleri indexlemek için (332.036 belge)
        create_and_index_documents(documents=documents_list, batch_size=1000)
    else:
        print("Belge yüklenemedi. İşlem sonlandırıldı.")