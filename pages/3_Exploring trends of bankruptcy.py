import streamlit as st
from For_streamlit import df, numeric_df
import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from page_functions import col_analyse
from scipy.stats import zscore, levene, ttest_ind, mannwhitneyu
import plotly.graph_objects as go

sorted_cols_lst = []


def ftu_tests(col):
    data_sum = df[col].sum()
    if data_sum >= 10000:
        filtered_col = df[df[col] > 1][col]
    elif (df[col] == 1).any() or (df[col] == 0).any():
        filtered_col = df[(df[col] > 0) & (df[col] < 1)][col]
    else:
        filtered_col = df[col]

    # Группировка данных
    bankrupt_group = filtered_col[df['Bankrupt?'] == 1]
    non_bankrupt_group = filtered_col[df['Bankrupt?'] == 0]

    # Проверка пустых данных
    if bankrupt_group.empty or non_bankrupt_group.empty:
        return

    # Levene Test
    f_stat, fp_value = levene(bankrupt_group, non_bankrupt_group)
    if fp_value >= 0.05:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        if tp_value >= 0.05:
            return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}'
        else:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'
    else:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        if tp_value >= 0.05:
            return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}'
        else:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'






def sort_by_ftu_tests(col):
    data_sum = df[col].sum()
    if data_sum >= 10000:
        filtered_col = df[df[col] > 1][col]
    elif (df[col] == 1).any() or (df[col] == 0).any():
        filtered_col = df[(df[col] > 0) & (df[col] < 1)][col]
    else:
        filtered_col = df[col]

    bankrupt_group = filtered_col[df['Bankrupt?'] == 1]
    non_bankrupt_group = filtered_col[df['Bankrupt?'] == 0]

    if bankrupt_group.empty or non_bankrupt_group.empty:
        return

    f_stat, fp_value = levene(bankrupt_group, non_bankrupt_group)
    if fp_value >= 0.05:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                sorted_cols_lst.append(col)
    else:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                sorted_cols_lst.append(col)



for col in numeric_df.columns:
    sort_by_ftu_tests(col)

ftu_df = pd.concat([df['Bankrupt?'], df[sorted_cols_lst]], axis=1)
num_ftu_df = df[sorted_cols_lst]


def filter_by_category(choice):
    if choice == 'Both':
        filtered_df = ftu_df
    elif choice == 'Bankrupt':
        filtered_df = ftu_df[ftu_df['Bankrupt?'] == 1]
    else:  # Non-Bankrupt
        filtered_df = ftu_df[ftu_df['Bankrupt?'] == 0]

    return filtered_df


bankrupt_button = st.sidebar.radio('Which category to show:', options=['Both', 'Bankrupt', 'Non-Bankrupt'])

filtered_df = filter_by_category(bankrupt_button)

st.markdown('### Now let\'s look on the dataset with indicators which have significant statistical differences between the bankrupt and non-bankrupt groups!')

st.dataframe(filtered_df, use_container_width=True)
st.write(len(sorted_cols_lst))


# st.write("Columns in cols_lst:", cols_lst)
# st.write("Columns in df:", df.columns)
# st.write("ftu_df columns:", ftu_df.columns)
# st.write("num_ftu_df columns:", num_ftu_df.columns)


def corr_column(col):
    # Проверяем, есть ли столбец в данных
    if col not in num_ftu_df.columns:
        return f"Column {col} not found in num_ftu_df."

    # Сумма значений в столбце
    data_sum = num_ftu_df[col].sum()

    # Фильтрация по значениям
    if data_sum >= 10000:
        filtered_col = num_ftu_df[num_ftu_df[col] > 1][col]
    elif (num_ftu_df[col] == 1).any() or (num_ftu_df[col] == 0).any():
        filtered_col = num_ftu_df[(num_ftu_df[col] > 0) & (num_ftu_df[col] < 1)][col]
    else:
        filtered_col = num_ftu_df[col]

    # Проверяем, не пустой ли фильтрованный столбец
    if filtered_col.empty:
        return "No data available for correlation."

    # Вычисляем корреляции
    correlations = num_ftu_df.corr()[col]
    high_correlations = correlations[correlations > 0.7].dropna()
    high_correlations = high_correlations.drop(col, errors='ignore')

    # Формируем текст результата
    if high_correlations.empty:
        return "No correlations with the table."
    else:
        return "The column has high correlations with:<br>" + "<br>".join(
            [f"{col}: {value:.2f}" for col, value in high_correlations.items()]
        )




def bankrupt_disc_column(column):
    column = ftu_df[column][ftu_df['Bankrupt?'] == 1]

    if column.sum() >= 10000:
        filtered_col = column[column > 1]
    elif (column == 1).any() or (column == 0).any():
        filtered_col = column[(column > 0) & (column < 1)]
    else:
        filtered_col = column

    filtered_col = filtered_col.drop_duplicates()
    bankrupt_stats = filtered_col.describe()
    bankrupt_text = "<br>Bankrupts:<br><br>" + "<br>".join([f"{key}: {value:.3f}" for key, value in bankrupt_stats.to_dict().items()])

    return bankrupt_text

def non_bankrupt_disc_column(column):
    column = ftu_df[column][ftu_df['Bankrupt?'] == 0]

    if column.sum() >= 10000:
        filtered_col = column[column > 1]
    elif (column == 1).any() or (column == 0).any():
        filtered_col = column[(column > 0) & (column < 1)]
    else:
        filtered_col = column

    filtered_col = filtered_col.drop_duplicates()
    non_bankrupt_stats = filtered_col.describe()
    non_bankrupt_text = "Non-Bankrupts:<br><br>" + "<br>".join(
        [f"{key}: {value:.3f}" for key, value in non_bankrupt_stats.items()]
    )

    return non_bankrupt_text


if bankrupt_button == 'Both':
    wrapper_name = "Columns of statistical significant differences for bankrupts and non-bankrupts"

    if not sorted_cols_lst:
        st.error("No significant columns found.")
    else:
        # Инициализация списков для данных
        features, categories, subcategories = [], [], []
        values = []
        details_category = []

        for col in sorted_cols_lst:
            corr_text = corr_column(col)  # Корреляционный текст
            ftu_result = ftu_tests(col)  # Результаты теста

            # Данные для верхнего уровня (Feature)
            features.extend([wrapper_name] * 3)  # Привязка к категории
            categories.extend([col] * 3)  # Колонка как категория
            subcategories.extend([
                corr_text,
                bankrupt_disc_column(col),
                non_bankrupt_disc_column(col)
            ])
            values.extend([0.5, 0.25, 0.25])  # Пропорции: 50% категория, 25% подкатегории

            # Добавляем детализированные данные для каждого уровня
            details_category.extend([f"<br>{ftu_result}"] * 3)

        # Создаем DataFrame с деталями
        tree_data = pd.DataFrame({
            "Feature": features,
            "Category": categories,
            "Subcategory": subcategories,
            "Value": values,
            "Column": details_category
        })

        # Построение Treemap
        fig = px.treemap(
            tree_data,
            path=["Feature", "Category", "Subcategory"],  # Иерархия уровней
            values="Value",  # Размер узлов
            custom_data=["Column"]  # Доп. данные для hover
        )

        # Настройка hovertemplate для разных уровней
        fig.update_traces(
            root_color="lightblue",  # Цвет корневого узла
            marker=dict(line=dict(width=2, color="DarkSlateGray")),  # Стиль линий
            hovertemplate="<b>%{label}</b><br>Values of significant statistical differences between bankrupts and non-bankrupts by test: %{customdata[0]}"  # Формат подсказок
        )

        # Отображение графика
        st.plotly_chart(fig, use_container_width=True)






#
#
# st.write(len(num_ftu_df.columns))
#
#
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(df.select_dtypes(include='number'))
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(scaled_data)
#
# df['PCA1'] = pca_result[:, 0]
# df['PCA2'] = pca_result[:, 1]
#
# st.markdown("### PCA Analysis")
# st.markdown("This scatter plot shows the data reduced to two principal components.")
# fig, ax = plt.subplots()
# sns.scatterplot(x='PCA1', y='PCA2', hue='Bankrupt?', data=df, ax=ax)
# st.pyplot(fig)



