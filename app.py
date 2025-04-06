import streamlit as st
import pandas as pd
import pickle
import json
import requests
import ast

def create_indices(df):
    return pd.Series(df.index, index=df['title']).drop_duplicates()

# ========== Загрузка сериализованных данных ==========

def load_data():
    status = st.empty()  # Создаем пустой контейнер для обновления текста

    status.text("🔹 Загружаем preprocessed_movies.csv...")
    df = pd.read_csv('preprocessed_movies.csv')
    indices = create_indices(df)

    status.text("🔹 Загружаем cosine_sim_tfidf.pkl...")
    with open('cosine_sim_tfidf.pkl', 'rb') as f:
        cosine_sim_tfidf = pickle.load(f)

    status.text("🔹 Загружаем cosine_sim_count.pkl...")
    with open('cosine_sim_count.pkl', 'rb') as f:
        cosine_sim_count = pickle.load(f)

    status.success("✅ Все данные успешно загружены.")  # Обновляем текст на успешное завершение
    return df, cosine_sim_tfidf, cosine_sim_count, indices

# ========== Функция для получения рекомендаций ==========

def get_recommendations(title, cosine_sim, indices, df, n=10):
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices].tolist()
    except KeyError:
        return None  # Возвращаем None, если фильм не найден

# ========== Интерфейс Streamlit ==========

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎥 Рекомендательная система фильмов</h1>", unsafe_allow_html=True)
st.write("Введите название фильма, и получите рекомендации 🎯")

# OMDb API ключ
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
def get_poster(imdbid):
    """Получить URL постера фильма по imdbid через TMDb API."""
    url = f"https://api.themoviedb.org/3/find/{imdbid}?api_key={TMDB_API_KEY}&external_source=imdb_id"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['movie_results']:
            poster_path = data['movie_results'][0].get('poster_path')
            if poster_path:
                base_url = "https://image.tmdb.org/t/p/w500"
                return f"{base_url}{poster_path}"  # Вернуть полный URL постера
    return None  # Вернуть None, если постер не найден

# Загружаем данные
with st.spinner("Загрузка данных..."):
    df, sim_tfidf, sim_count, title_indices = load_data()

# Поле ввода
movie_name = st.text_input("Название фильма", "", placeholder="Например: Inception")

# Слайдер — количество рекомендаций
top_n = st.slider("Сколько фильмов показать?", min_value=5, max_value=20, value=10)

# Добавляем выбор метода
method = st.radio(
    "Выберите метод для рекомендаций:",
    options=["TF-IDF", "CountVectorizer"]
)

df['year'] = df['release_date'].str[:4]  # Извлекаем год из даты

# Обработка запроса
if movie_name:
    with st.spinner("Ищем похожие фильмы..."):
        if method == "TF-IDF":
            recommendations = get_recommendations(movie_name, sim_tfidf, title_indices, df, n=top_n)
        else:
            recommendations = get_recommendations(movie_name, sim_count, title_indices, df, n=top_n)

    if recommendations:
        st.subheader(f"🎯 Рекомендации ({method}) для: {movie_name}")
        
        # Создаем колонки для фильмов
        cols = st.columns(len(recommendations))  # Создаем столько колонок, сколько фильмов
        
        for idx, rec in enumerate(recommendations):
            with cols[idx]:  # Размещаем каждый фильм в своей колонке
                # Получаем строку DataFrame для текущего фильма
                movie_row = df[df['title'] == rec]
                if not movie_row.empty:
                    imdbid = movie_row.iloc[0]['imdb_id']  # Получаем imdbid
                    poster_url = get_poster(imdbid)  # Получаем URL постера
                    if poster_url:
                        st.image(poster_url, width=1000)  # Отображаем постер
                    else:
                        st.write("Постер недоступен")
                    
                    # Получаем режиссера и актеров из столбцов
                    director = movie_row.iloc[0]['director']  # Берем режиссера из столбца
                    actors = movie_row.iloc[0]['actors']  # Берем строку актеров из столбца

                    # Преобразуем строку в список
                    try:
                        actors = ast.literal_eval(actors)
                    except (ValueError, SyntaxError):
                        actors = []  # Если преобразование не удалось, задаем пустой список

                    # Убедимся, что actors — это список строк
                    if not isinstance(actors, list):
                        actors = []
                    actors = [str(actor).strip() for actor in actors]  # Убираем лишние пробелы

                    st.markdown(f"<p style='font-size:20px; font-family:Merriweather, serif; font-weight:bold;'>{rec}</p>", unsafe_allow_html=True)
                    st.markdown(f"**Год:** {movie_row.iloc[0]['year']}")  # Год под названием
                    st.markdown(f"**Режиссер:** <br> {director}", unsafe_allow_html=True)
                    st.markdown(f"**Актеры:** <br> {'<br>'.join(actors)}", unsafe_allow_html=True)
                    st.markdown(poster_url)
    else:
        st.warning("Некорректное название фильма (такого нет в базе).")