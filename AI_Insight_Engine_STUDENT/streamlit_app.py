import io
import re
import random
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =====================================================================
# 1. 핵심 분석 함수들
# =====================================================================
def read_and_clean_csv(file_path):
    encodings = ['utf-8-sig', 'utf-8', 'cp949']
    df_raw = None
    for enc in encodings:
        try:
            df_raw = pd.read_csv(file_path, encoding=enc)
            break
        except:
            continue
            
    if df_raw is None:
        raise ValueError("지원되는 인코딩으로 파일을 열 수 없습니다.")
    if 'text' not in df_raw.columns:
        raise ValueError("'text' 컬럼이 존재하지 않습니다.")
        
    df = df_raw.copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df.dropna(subset=['text'])
    df = df[df['text'] != '']
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    return df

def get_cluster_keywords(df, top_n=6):
    stopwords = {"청년", "지역", "광주", "전남", "정보", "경우", "부분", "요즘", "실제로", "생각합니다", "좋겠습니다", "어렵다", "어렵습니다", "필요하다", "필요합니다", "있으면"}
    grouped = df.groupby('cluster')['text'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[가-힣]{2,}\b', ngram_range=(1, 2), stop_words=list(stopwords))
    
    tfidf_matrix = vectorizer.fit_transform(grouped['text'])
    feature_names = vectorizer.get_feature_names_out()
    
    result_data = []
    for i in range(tfidf_matrix.shape[0]):
        cluster_id = grouped['cluster'].iloc[i]
        row_scores = tfidf_matrix.getrow(i).toarray().flatten()
        top_indices = row_scores.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices if row_scores[idx] > 0]
        result_data.append({'cluster': cluster_id, 'keywords': ", ".join(top_keywords)})
    return pd.DataFrame(result_data)

def get_closest_sentences(df, embeddings, kmeans, top_n=1):
    results = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_idx = df[df['cluster'] == cluster_id].index
        if len(cluster_idx) == 0: continue
            
        cluster_embs = embeddings[cluster_idx]
        center_vec = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
        similarities = cosine_similarity(cluster_embs, center_vec).flatten()
        
        best_local_idx = similarities.argsort()[-1]
        results.append({
            'cluster': cluster_id,
            '대표 의견': df.loc[cluster_idx[best_local_idx], 'text']
        })
    return pd.DataFrame(results)

def visualize_clusters_2d(df, embeddings):
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1],
        'cluster': df['cluster'].astype(str), 'text': df['text']
    })
    
    fig = px.scatter(
        plot_df, x='x', y='y', color='cluster',
        hover_data={'x': False, 'y': False, 'cluster': True, 'text': True},
        title="Topic Map (PCA 2D)", opacity=0.7
    )
    fig.update_layout(width=900, height=500, plot_bgcolor='white', margin=dict(l=20, r=20, t=40, b=20))
    return fig

# =====================================================================
# 2. Gradio 인터페이스용 Wrapper 함수
# =====================================================================
def gradio_analyze(file_path, k):
    if file_path is None:
        return "파일을 업로드해주세요.", None, None, None
        
    try:
        # 1. 정제
        df = read_and_clean_csv(file_path)
        
        # 2. 임베딩
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(df['text'].tolist(), show_progress_bar=False)
        
        # 3. 클러스터링
        kmeans = KMeans(n_clusters=int(k), random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(embeddings)
        
        # 4. 요약표 생성
        count_df = df.groupby('cluster').size().reset_index(name='의견 수')
        kw_df = get_cluster_keywords(df)
        rep_df = get_closest_sentences(df, embeddings, kmeans)
        
        summary_table = pd.merge(count_df, kw_df, on='cluster')
        summary_table = pd.merge(summary_table, rep_df, on='cluster')
        
        # 5. 토픽맵 생성
        fig = visualize_clusters_2d(df, embeddings)
        
        # 상태 저장을 위한 딕셔너리
        state = {
            "df": df,
            "embeddings": embeddings,
            "model": model
        }
        
        count_str = f"총 {len(df):,}개의 의견이 분석되었습니다."
        return count_str, summary_table, fig, state
        
    except Exception as e:
        return f"오류 발생: {str(e)}", None, None, None

def gradio_search(query, top_k, state):
    if state is None or "model" not in state:
        return pd.DataFrame({"알림": ["먼저 분석(Analyze)을 실행해주세요."]})
    if not query.strip():
        return pd.DataFrame({"알림": ["검색어를 입력해주세요."]})
        
    df = state["df"]
    embeddings = state["embeddings"]
    model = state["model"]
    
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = similarities.argsort()[-int(top_k):][::-1]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            'Rank': rank,
            'Score': round(similarities[idx], 4),
            'Cluster': df.iloc[idx]['cluster'],
            'Text': df.iloc[idx]['text']
        })
        
    return pd.DataFrame(results)

# =====================================================================
# 3. Gradio UI 레이아웃 구성 (Blocks)
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 텍스트 군집 분석 & 시맨틱 검색 대시보드")
    
    # 세션 동안 임베딩과 모델을 유지하기 위한 전역 상태 변수
    app_state = gr.State()
    
    with gr.Row():
        # 좌측: 입력부
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 1. 입력 설정")
            file_input = gr.File(label="CSV Upload", type="filepath")
            k_input = gr.Slider(minimum=2, maximum=20, step=1, value=5, label="Number of Topics (k)")
            analyze_btn = gr.Button("Analyze 🚀", variant="primary")
            
            gr.Markdown("### 📈 분석 결과")
            count_output = gr.Textbox(label="분석된 의견 수", interactive=False)
            
        # 우측: 출력부 (Summary)
        with gr.Column(scale=2):
            gr.Markdown("### 📋 2. Topic Summary")
            summary_output = gr.Dataframe(label="클러스터별 요약", interactive=False, wrap=True)
            
    # 하단: 시각화부 (Topic Map)
    gr.Markdown("### 🗺️ 3. Topic Map")
    plot_output = gr.Plot(label="2D 군집 시각화")
    
    gr.Markdown("---")
    
    # 최하단: 검색부
    gr.Markdown("### 🔍 4. Semantic Search")
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(label="Query", placeholder="검색할 키워드나 문장을 입력하세요...")
        with gr.Column(scale=1):
            top_k_input = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Top-K")
        with gr.Column(scale=1):
            search_btn = gr.Button("Search", variant="secondary")
            
    search_output = gr.Dataframe(label="Semantic Search Result", interactive=False, wrap=True)

    # =================================================================
    # 4. 버튼 이벤트 연결 (인풋 -> 함수 -> 아웃풋)
    # =================================================================
    analyze_btn.click(
        fn=gradio_analyze,
        inputs=[file_input, k_input],
        outputs=[count_output, summary_output, plot_output, app_state]
    )
    
    search_btn.click(
        fn=gradio_search,
        inputs=[query_input, top_k_input, app_state],
        outputs=[search_output]
    )

# 앱 실행 (배포 환경을 위해 파라미터 제외)
if __name__ == "__main__":
    app.launch(share=True)
