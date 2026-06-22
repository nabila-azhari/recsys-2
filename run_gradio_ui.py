import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

print("=" * 60)
print("   MENYIAPKAN ANTARMUKA (UI) INTERAKTIF RECOMMENDER SYSTEM")
print("=" * 60)

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device yang digunakan: {device.upper()}")
if device == "cuda":
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# 2. Setup Dataset URL
BASE_URL = "https://raw.githubusercontent.com/nabila-azhari/recsys-2/master/data"
train_news_url = f"{BASE_URL}/MINDsmall_train/MINDsmall_train/news.tsv"
train_behaviors_url = f"{BASE_URL}/MINDsmall_train/MINDsmall_train/behaviors.tsv"
dev_news_url = f"{BASE_URL}/MINDsmall_dev/MINDsmall_dev/news.tsv"
dev_behaviors_url = f"{BASE_URL}/MINDsmall_dev/MINDsmall_dev/behaviors.tsv"

news_cols = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
behaviors_cols = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']

print("\n[Langkah 1/5] Mengunduh dataset MIND-small dari GitHub (Mohon tunggu)...")
try:
    df_train_news = pd.read_csv(train_news_url, sep='\t', names=news_cols, header=None)
    df_train_behaviors = pd.read_csv(train_behaviors_url, sep='\t', names=behaviors_cols, header=None)
    df_dev_news = pd.read_csv(dev_news_url, sep='\t', names=news_cols, header=None)
    print("Dataset berhasil diunduh!")
except Exception as e:
    print(f"Gagal memuat dari branch master: {e}. Mencoba fallback ke branch main...")
    ALT_BASE_URL = "https://raw.githubusercontent.com/nabila-azhari/recsys-2/main/data"
    train_news_url = f"{ALT_BASE_URL}/MINDsmall_train/MINDsmall_train/news.tsv"
    train_behaviors_url = f"{ALT_BASE_URL}/MINDsmall_train/MINDsmall_train/behaviors.tsv"
    dev_news_url = f"{ALT_BASE_URL}/MINDsmall_dev/MINDsmall_dev/news.tsv"
    
    df_train_news = pd.read_csv(train_news_url, sep='\t', names=news_cols, header=None)
    df_train_behaviors = pd.read_csv(train_behaviors_url, sep='\t', names=behaviors_cols, header=None)
    df_dev_news = pd.read_csv(dev_news_url, sep='\t', names=news_cols, header=None)
    print("Dataset berhasil diunduh via branch main!")

# 3. Preprocessing
print("\n[Langkah 2/5] Melakukan preprocessing data...")
def preprocess_news(df):
    df_cleaned = df.copy()
    df_cleaned['Title'] = df_cleaned['Title'].fillna("")
    df_cleaned['Abstract'] = df_cleaned['Abstract'].fillna("")
    df_cleaned['content'] = df_cleaned['Title'] + " " + df_cleaned['Abstract']
    df_cleaned = df_cleaned.drop_duplicates(subset=['NewsID']).reset_index(drop=True)
    return df_cleaned

df_train_news_processed = preprocess_news(df_train_news)
df_dev_news_processed = preprocess_news(df_dev_news)

# Corpus gabungan untuk metadata berita
all_news = pd.concat([df_train_news_processed, df_dev_news_processed]).drop_duplicates(subset=['NewsID']).reset_index(drop=True)
news_metadata = all_news.set_index('NewsID')

# 4. SBERT Embeddings
print("\n[Langkah 3/5] Memuat model Sentence-BERT & membuat embedding berita...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

news_contents = all_news['content'].tolist()
news_ids = all_news['NewsID'].tolist()

embeddings = model.encode(
    news_contents, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True
)
news_embedding_dict = {news_ids[i]: embeddings[i] for i in range(len(news_ids))}
print(f"Embedding berhasil dibuat! Total berita: {len(news_embedding_dict)}")

# 5. Recommendation Core Logic
print("\n[Langkah 4/5] Mempersiapkan logika recommendation engine...")

def build_user_profile(user_id, df_behaviors, embedding_dict):
    user_rows = df_behaviors[df_behaviors['UserID'] == user_id]
    if user_rows.empty:
        return None
        
    history_news_ids = []
    for history_str in user_rows['History'].dropna():
        history_news_ids.extend(history_str.split())
        
    history_news_ids = list(set(history_news_ids))
    
    valid_embeddings = []
    for news_id in history_news_ids:
        if news_id in embedding_dict:
            valid_embeddings.append(embedding_dict[news_id])
            
    if not valid_embeddings:
        return None
        
    return np.mean(valid_embeddings, axis=0)

def recommend_news(user_id, df_behaviors, embedding_dict, metadata_df, top_k=10):
    user_profile = build_user_profile(user_id, df_behaviors, embedding_dict)
    if user_profile is None:
        return f"Error: User ID '{user_id}' tidak memiliki riwayat klik (History) di dalam database latihan."
        
    # Filter berita yang sudah pernah dibaca
    user_rows = df_behaviors[df_behaviors['UserID'] == user_id]
    read_news_ids = set()
    for history_str in user_rows['History'].dropna():
        read_news_ids.update(history_str.split())
        
    candidate_news_ids = []
    candidate_embeddings = []
    
    for news_id, emb in embedding_dict.items():
        if news_id not in read_news_ids:
            candidate_news_ids.append(news_id)
            candidate_embeddings.append(emb)
            
    if not candidate_embeddings:
        return "Error: Tidak ada berita kandidat baru yang tersedia."
        
    candidate_embeddings = np.array(candidate_embeddings)
    user_profile_reshaped = user_profile.reshape(1, -1)
    similarities = cosine_similarity(user_profile_reshaped, candidate_embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        news_id = candidate_news_ids[idx]
        score = similarities[idx]
        
        title = metadata_df.loc[news_id, 'Title'] if news_id in metadata_df.index else "Tidak ada Judul"
        category = metadata_df.loc[news_id, 'Category'] if news_id in metadata_df.index else "Lainnya"
        
        recommendations.append({
            'Rank': rank,
            'NewsID': news_id,
            'Title': title,
            'Category': category,
            'Similarity Score': round(float(score), 4)
        })
        
    return pd.DataFrame(recommendations)

# wrapper for Gradio
def gradio_recommend(user_id, top_k):
    user_id = str(user_id).strip()
    if not user_id:
        return gr.update(visible=False), "Mohon masukkan User ID terlebih dahulu."
        
    res = recommend_news(
        user_id=user_id,
        df_behaviors=df_train_behaviors,
        embedding_dict=news_embedding_dict,
        metadata_df=news_metadata,
        top_k=int(top_k)
    )
    
    if isinstance(res, str):
        return gr.update(visible=False), res
        
    return gr.update(value=res, visible=True), f"Menampilkan {top_k} rekomendasi berita untuk User ID: {user_id}"

# 6. Gradio Interface Construction
print("\n[Langkah 5/5] Membuat web antarmuka dengan Gradio...")

# Clean and minimalist Base Theme
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown(
        """
        # Sistem Rekomendasi Berita
        Sistem ini memberikan rekomendasi berita personal berdasarkan riwayat membaca pengguna.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(
                label="User ID",
                placeholder="Masukkan ID Pengguna...",
                value="U13740"
            )
            top_k_slider = gr.Slider(
                label="Jumlah Berita (K)",
                minimum=3,
                maximum=20,
                step=1,
                value=10
            )
            btn = gr.Button("Tampilkan Rekomendasi", variant="primary")
            
            gr.Markdown(
                """
                **Contoh User ID Aktif:**
                * U13740
                * U91836
                * U84444
                """
            )
            
        with gr.Column(scale=2):
            status_output = gr.Markdown("Masukkan User ID di kolom kiri lalu klik tombol untuk melihat rekomendasi.")
            results_table = gr.Dataframe(
                headers=["Rank", "NewsID", "Title", "Category", "Similarity Score"],
                datatype=["str", "str", "str", "str", "number"],
                visible=False
            )
            
    btn.click(
        fn=gradio_recommend,
        inputs=[user_input, top_k_slider],
        outputs=[results_table, status_output]
    )

print("\nUI SIAP! Membuka server lokal dan meluncurkan browser...")
# inbrowser=True will automatically open their local web browser!
demo.launch(inbrowser=True)
