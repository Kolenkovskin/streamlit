import streamlit as st
from For_streamlit import df, numeric_df
from page_functions import col_analyse

head = 'What is the deal with?'
disc = 'The dataset used in this project comprises financial indicators that are critical in assessing a company\'s financial stability, performance, and risk of bankruptcy. These indicators are not only vital for evaluating a company\'s health but also provide key insights into various business areas and sectors. These indicators are indispensable tools for equity analysts, portfolio managers, and investors who need to evaluate the financial health of companies before making investment decisions. Creditors, including banks and financial institutions, use these indicators to assess a company’s creditworthiness and risk of insolvency. Management teams rely on these metrics for strategic planning, operational improvements, and decision-making, especially in competitive markets. In M&A activities, these metrics play a key role in due diligence processes to determine the valuation and feasibility of acquisitions. By understanding these indicators and their applications, stakeholders can make informed decisions, navigate market challenges, and seize opportunities for growth and profitability.'

st.header(head)  # Заголовок таблицы
st.write(disc)   # Описание таблицы

st.divider()

# Настройка selectbox с пустым значением по умолчанию
selected_column = st.sidebar.selectbox(
    'Choose a column to display:',
    options=['Select a column...'] + list(numeric_df.columns),  # Добавляем опцию "Select a column..."
    index=0  # Устанавливаем первую опцию как выбранную по умолчанию
)


if selected_column == 'Select a column...':
    st.subheader("Choose a column and let's get started!")

else:
    col_analyse(selected_column)








