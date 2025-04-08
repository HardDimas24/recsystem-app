# 🎬 Рекомендательная система фильмов

Веб-приложение на Streamlit для получения персонализированных рекомендаций фильмов на основе схожести контента. Система использует два метода анализа: TF-IDF и CountVectorizer.

## Ссылка на сайт

https://recsystem-app-hd24.streamlit.app/

## Основные возможности

- 🔍 Поиск фильмов по названию
- 🎯 Получение рекомендаций похожих фильмов
- 🔄 Выбор метода рекомендаций (TF-IDF или CountVectorizer)
- 🎚️ Настройка количества рекомендаций (5-20 фильмов)
- 🖼️ Отображение постеров фильмов через TMDb API
- 📅 Информация о годе выпуска, режиссере и актерах

## Струтура проекта
```
movie-recommender/
├── app.py                     
├── preprocessed_movies.csv    
├── cosine_sim_tfidf.pkl      
├── cosine_sim_count.pkl       
├── requirements.txt           
├── .devcontainer/           
├── .gitattributes         
├── .gitignore            
└── LICENSE
```

### Описание файлов:

#### 🎯 Основной файл
- `app.py` - включает в себя загрузку данных и визуаолизацию

#### 📊 Файлы данных
- `preprocessed_movies.csv` - обработанный датасет фильмов с метаданными (9216 фильмов)
- `cosine_sim_tfidf.pkl` и `cosine_sim_count.pkl` - предрассчитанные матрицы сходства для двух методов рекомендаций (отслеживаются через Git LFS из-за большого размера (где-то 700мб))

#### ⚙️ Конфигурационные файлы
- `.devcontainer/` - настройки для разработки в контейнере
- `.gitattributes` - настройки Git LFS
- `.gitignore` - список игнорируемых файлов
- `requirements.txt` - зависимости Python

#### 📜 `LICENSE`
Лицензионное соглашение проекта

## Скриншоты
<img width="1399" alt="image" src="https://github.com/user-attachments/assets/f3606622-ecb3-4223-9836-c8327cbc66bb" />
<p><em>Главное меню</em></p>

<img width="1399" alt="image" src="https://github.com/user-attachments/assets/0b3c4ebb-a561-4e6f-ad27-c081b00ab844" />
<p><em>Результат работы</em></p>

<img width="1399" alt="image" src="https://github.com/user-attachments/assets/e20fde9f-1fda-4e71-be9a-872a39c49392" />
<p><em>Проверка на корректность названия</em></p>
