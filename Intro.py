import streamlit as st
from For_streamlit import df, exploring_header, intro, numeric_df

st.set_page_config('Financial data analyse.', layout='wide')

st.header('Introduction.', divider='grey')

col1, col2 = st.columns(2)
col1.write(intro)
col2.image('DALL·E 2024-11-19 21.05.01 - A visually striking and professional illustration symbolizing financial analysis, featuring graphs, charts, and data visualizations. The scene include.webp', caption='Explore and analyse')

st.divider()

df_head = 'Exploring data'
df_disc = 'The dataset used in this project contains financial indicators for companies, with a primary focus on identifying bankruptcy status. It includes a diverse range of features that highlight the financial health and operational metrics of the organizations. The dataset provides a comprehensive view of financial stability and is instrumental in analyzing key differences between bankrupt and non-bankrupt companies. It also allows for deeper exploration of statistical relationships and the evaluation of significant indicators impacting financial outcomes. In this Streamlit project, the dataset is presented interactively to explore the financial metrics and uncover insights. Users can filter data by bankruptcy status and visualize trends that distinguish stable companies from those at risk. This analysis highlights critical financial patterns, including profitability, asset utilization, and cash flow management, providing valuable insights for decision-making.'

st.subheader(df_head)  # Заголовок таблицы
st.write(df_disc)   # Описание таблицы

st.dataframe(df, use_container_width=True)








