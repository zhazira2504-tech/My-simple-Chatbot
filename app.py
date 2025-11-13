# -*- coding: utf-8 -*-
import streamlit as st
from transformers import pipeline

# Используем декоратор Streamlit для кеширования.
# Эта функция выполнится только один раз, при первом запуске приложения.
# Результат (загруженная модель) сохранится в памяти.
@st.cache_resource
def load_model():
    print("Загрузка модели...")
    generator = pipeline('text-generation', model='sberbank-ai/rugpt3small_based_on_gpt2')
    print("Модель загружена.")
    return generator

def generate_response(prompt, generator):
    """
    Функция генерации ответа. Она почти не изменилась.
    """
    generated = generator(
        prompt,
        max_length=len(prompt.encode('utf-8')) + 60,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=40
    )
    full_text = generated[0]['generated_text']
    bot_response = full_text[len(prompt):].strip()
    return bot_response

# --- Интерфейс нашего веб-приложения ---

st.title("🤖 Мой Умный Облачный Чат-бот")
st.write("Этот бот использует нейросеть ruGPT-3 Small для генерации ответов.")

# Загружаем модель с помощью нашей кешированной функции
generator = load_model()

# Инициализируем "память" бота (историю диалога)
# st.session_state - это специальный объект Streamlit, который не сбрасывается
# при взаимодействиях пользователя.
if 'history' not in st.session_state:
    st.session_state['history'] = ""

# Выводим историю чата на экран
if st.session_state.history:
    st.write("**История диалога:**")
    st.write(st.session_state.history.replace("\n", "<br>"), unsafe_allow_html=True)
    st.write("---")

# Создаем поле для ввода
user_text = st.text_input("Ваше сообщение:", key="input")

# Кнопка для отправки
if st.button("Отправить"):
    if user_text:
        # Формируем промпт, добавляя историю
        prompt = st.session_state.history + f"Вы: {user_text}\nБот:"

        # Генерируем ответ
        with st.spinner("Бот думает..."): # Показываем индикатор загрузки
            bot_response = generate_response(prompt, generator)

        # Обновляем историю
        st.session_state.history += f"Вы: {user_text}\nБот: {bot_response}\n"

        # "Очищаем" поле ввода, чтобы было удобнее писать следующее сообщение
        st.rerun()
    else:
        st.warning("Пожалуйста, введите сообщение.")
