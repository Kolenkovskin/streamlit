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

    f_stat, fp_value = levene(bankrupt_group, non_bankrupt_group)
    if fp_value >= 0.05:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'
    else:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'

    # # Levene Test
    # f_stat, fp_value = levene(bankrupt_group, non_bankrupt_group)
    # if fp_value >= 0.05:
    #     t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
    #     if tp_value >= 0.05:
    #         return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}'
    #     else:
    #         u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
    #         return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'
    # else:
    #     t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
    #     if tp_value >= 0.05:
    #         return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}'
    #     else:
    #         u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
    #         return f'F-Statistic: {f_stat}, P-value: {fp_value}<br>T-Statistic: {t_stat}, P-value: {tp_value}<br>U-Statistic: {u_stat}, P-value: {up_value}'






def sort_by_ftu_tests(column):
    data = df[column].drop_duplicates()
    if data.sum() >= 10000:
        filtered_column = data[data > 1]
    elif data.any() == 1 or data.any() == 0:
        filtered_column = data[(data > 0) & (data < 1)]
    else:
        filtered_column = data

    bankrupt_group = filtered_column[df['Bankrupt?'] == 1]
    non_bankrupt_group = filtered_column[df['Bankrupt?'] == 0]

    if bankrupt_group.empty or non_bankrupt_group.empty:
        return

    f_stat, fp_value = levene(bankrupt_group, non_bankrupt_group)
    if fp_value >= 0.05:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                sorted_cols_lst.append(column)
    else:
        t_stat, tp_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        if tp_value < 0.05:
            u_stat, up_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if up_value < 0.05:
                sorted_cols_lst.append(column)



for col in numeric_df.columns:
    sort_by_ftu_tests(col)

num_ftu_df = df[sorted_cols_lst]
ftu_df = pd.concat([df['Bankrupt?'], num_ftu_df], axis=1)



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



def corr_column(column):
    data = num_ftu_df[column]

    if data.sum() >= 10000:
        filtered_col = data > 1
    elif data.any() == 1 or data.any() == 0:
        filtered_col = (data > 0) & (data < 1)
    else:
        filtered_col = data

    correlations = num_ftu_df.corr()[column]
    high_correlations = correlations[correlations > 0.7].dropna()
    high_correlations = high_correlations.drop(column, errors='ignore')

    # Формируем текст результата
    if high_correlations.empty:
        return "No correlations with the table."
    else:
        return "The column has high correlations with:<br><br>" + "<br>".join(
            [f"{column}: {value:.2f}" for column, value in high_correlations.items()]
        )

# def corr_column(column):
#     # Проверяем, есть ли столбец в данных
#     if column not in num_ftu_df.columns:
#         return f"Column {column} not found in num_ftu_df."
#
#     # Сумма значений в столбце
#     data_sum = num_ftu_df[column].sum()
#
#     # Фильтрация по значениям
#     if data_sum >= 10000:
#         filtered_col = num_ftu_df[num_ftu_df[column] > 1][column]
#     elif (num_ftu_df[column] == 1).any() or (num_ftu_df[column] == 0).any():
#         filtered_col = num_ftu_df[(num_ftu_df[column] > 0) & (num_ftu_df[column] < 1)][column]
#     else:
#         filtered_col = num_ftu_df[column]
#
#     # Проверяем, не пустой ли фильтрованный столбец
#     if filtered_col.empty:
#         return "No data available for correlation."
#
#     # Вычисляем корреляции
#     correlations = num_ftu_df.corr()[column]
#     high_correlations = correlations[correlations > 0.7].dropna()
#     high_correlations = high_correlations.drop(column, errors='ignore')
#
#     # Формируем текст результата
#     if high_correlations.empty:
#         return "No correlations with the table."
#     else:
#         return "The column has high correlations with:<br><br>" + "<br>".join(
#             [f"{column}: {value:.2f}" for column, value in high_correlations.items()]
#         )




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

    st.markdown(
        '### Now let\'s look on the dataset with columns which have significant statistical differences between the bankrupt and non-bankrupt groups!')

    st.dataframe(filtered_df, use_container_width=True)

    wrapper_name = "Columns of statistical significant differences for bankrupts and non-bankrupts"

    if not sorted_cols_lst:
        st.error("No significant columns found.")
    else:

        st.markdown('### In the graph below you can choose a column to discover it`s characteristics for bankrupt and non-bankrupt groups.')
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
            values.extend([0.6, 0.2, 0.2])  # Пропорции: 50% категория, 25% подкатегории

            # Добавляем детализированные данные для каждого уровня
            details_category.extend([f"<br>{ftu_result}"] * 3)

        # Создаем DataFrame с деталями
        tree_data = pd.DataFrame({
            "Feature": features,
            "Category": categories,
            "Subcategory": subcategories,
            "Value": values,
            "FTU Results": details_category,
            "Columns quantity": len(categories)
        })

        # Построение Treemap
        fig = px.treemap(
            tree_data,
            path=["Feature", "Category", "Subcategory"],  # Иерархия уровней
            values="Value",  # Размер узлов
            custom_data=["FTU Results", "Columns quantity"],  # Доп. данные для hover
            hover_data=None
        )

        # Настройка hovertemplate для разных уровней
        fig.update_traces(
            root_color="lightblue",  # Цвет корневого узла
            marker=dict(line=dict(width=2, color="DarkSlateGray")),  # Стиль линий
            hovertemplate="%{label}"
        )


        st.plotly_chart(fig, use_container_width=True)

        # Масштабируем данные
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(ftu_df.select_dtypes(include='number'))

        # Применяем PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # Добавляем компоненты обратно в DataFrame
        ftu_df['PCA1'] = pca_result[:, 0]
        ftu_df['PCA2'] = pca_result[:, 1]

        # Объяснённая дисперсия
        explained_variance = pca.explained_variance_ratio_ * 100

        # Создаём график
        st.markdown("### PCA Analysis")
        st.markdown(f"""
        This scatter plot shows the data reduced to two principal components:
        - **PCA1** explains {explained_variance[0]:.2f}% of the variance.
        - **PCA2** explains {explained_variance[1]:.2f}% of the variance.
        """)

        fig, ax = plt.subplots(figsize=(10, 6))  # Оптимальный размер графика
        sns.scatterplot(
            x='PCA1',
            y='PCA2',
            hue='Bankrupt?',
            palette='coolwarm',  # Используем мягкую палитру
            data=ftu_df,
            ax=ax,
            alpha=0.8,  # Прозрачность точек для устранения наложений
            edgecolor='k',  # Чёрная рамка вокруг точек для выделения
            s=80  # Размер точек
        )

        # Настройка заголовков и подписей
        ax.set_xlabel(f"PCA1 ({explained_variance[0]:.2f}% Variance)", fontsize=12)
        ax.set_ylabel(f"PCA2 ({explained_variance[1]:.2f}% Variance)", fontsize=12)
        ax.set_title("PCA Analysis: Principal Components Visualization", fontsize=16, fontweight='bold')
        ax.legend(title="Bankrupt?", fontsize=10, title_fontsize=12, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)  # Добавляем сетку для улучшения восприятия

        # Отображение
        st.pyplot(fig)

        st.markdown("""
        ### What does this graph represent?

        The **PCA Analysis: Principal Components Visualization** scatter plot provides a two-dimensional representation of the dataset after reducing its dimensionality using Principal Component Analysis (PCA).

        #### Key Insights:
        1. **Principal Components**:
           - **PCA1**: The first principal component, displayed on the X-axis, captures the largest variance in the data. It explains **24.30%** of the total variance.
           - **PCA2**: The second principal component, shown on the Y-axis, captures the second-largest variance orthogonal to PCA1. It explains **12.06%** of the variance.

        2. **Data Points**:
           - Each dot on the graph represents a data sample.
           - The **color of the dots** indicates whether the company was marked as "Bankrupt" (`1`, orange) or "Not Bankrupt" (`0`, blue).

        3. **Purpose**:
           - The graph helps visualize clusters, trends, or separations in the data between bankrupt and non-bankrupt companies, based on the two most significant components of variability.

        4. **Observations**:
           - Companies with similar financial characteristics tend to cluster together.
           - This visualization can guide further analysis on potential predictors of bankruptcy.

        This type of dimensionality reduction is particularly useful for understanding high-dimensional data in a simplified manner while retaining the most important information.
        """)

    st.subheader('Logistic regression')

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Подготовка данных
    X = num_ftu_df
    y = filtered_df['Bankrupt?']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Логистическая регрессия
    model = LogisticRegression()
    model.fit(X_scaled, y)
    coef = pd.Series(model.coef_[0], index=X.columns)

    fig, ax = plt.subplots(figsize=(14, 8))  # Увеличенный размер графика

    # Построение графика с цветовой дифференциацией
    coef.plot(
        kind='bar',
        ax=ax,
        color=(coef > 0).map({True: 'limegreen', False: 'tomato'}),
        edgecolor='black'  # Добавление чёрной рамки для контраста
    )

    # Настройка заголовка и меток осей
    ax.set_title("Logistic Regression Coefficients", fontsize=18, fontweight='bold')
    ax.set_xlabel("Features", fontsize=14, labelpad=15, fontstyle='italic')
    ax.set_ylabel("Coefficient Value", fontsize=14, labelpad=15)

    # Настройка сетки
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)

    # Ротация меток оси X
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Отображение значимых коэффициентов
    for idx, value in enumerate(coef):
        if abs(value) > 0.2:  # Порог значимости
            ax.text(
                idx,
                value + (0.02 if value > 0 else -0.02),
                f'{value:.2f}',
                ha='center',
                va='bottom' if value > 0 else 'top',
                fontsize=10,
                color='black'
            )

    # Добавление линии Y=0
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')

    # Улучшение отображения
    plt.tight_layout()

    # Отображение графика в Streamlit
    st.pyplot(fig)

    st.markdown("""
    ### What does this graph represent?

    The **Logistic Regression Coefficients** chart provides insights into how each financial feature contributes to the prediction of bankruptcy. 

    #### Key Elements:
    1. **Features**:
       - The X-axis lists all the financial indicators used in the logistic regression model.
       - Each bar represents the coefficient of the corresponding feature.

    2. **Coefficient Values**:
       - The Y-axis displays the magnitude and direction of each feature's influence on the bankruptcy prediction.
       - **Positive Coefficients (green bars)**: Indicate that an increase in the feature's value increases the likelihood of bankruptcy.
       - **Negative Coefficients (red bars)**: Indicate that an increase in the feature's value decreases the likelihood of bankruptcy.

    3. **Highlighted Coefficients**:
       - Only coefficients with significant influence (absolute value > 0.2) are labeled to emphasize their importance.

    4. **Purpose**:
       - This visualization helps identify the most influential financial indicators affecting bankruptcy predictions.
       - It allows users to focus on features with the strongest predictive power, aiding in financial decision-making and risk management.

    5. **Interpretation**:
       - Features with higher absolute coefficients have a stronger impact on the model's output.
       - Understanding these coefficients can help refine the logistic regression model and guide further data analysis.

    Usign this graph will help to identify and analyze key financial drivers of bankruptcy and improve your predictive modeling.
    """)

    from sklearn.cluster import KMeans

    # Нормализация данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_df.drop(columns=['Bankrupt?']))

    # Кластеризация
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    filtered_df['Cluster'] = clusters

    fig, ax = plt.subplots(figsize=(12, 8))  # Увеличиваем размер для лучшего восприятия

    # Построение кластерного анализа
    sns.scatterplot(
        x=filtered_df['PCA1'],
        y=filtered_df['PCA2'],
        hue=filtered_df['Cluster'],
        palette='coolwarm',  # Используем контрастную палитру
        style=filtered_df['Cluster'],  # Добавляем различие в маркерах
        size=filtered_df['PCA1'].abs(),  # Размер маркеров в зависимости от значений
        sizes=(50, 200),  # Диапазон размеров
        ax=ax,
        edgecolor='black',  # Добавляем черные рамки для четкости
        alpha=0.8  # Прозрачность для устранения наложения
    )

    # Настройка заголовка и подписей осей
    ax.set_title("Enhanced Cluster Analysis of Financial Data", fontsize=18, fontweight='bold', color='darkblue')
    ax.set_xlabel("PCA1 (24.30% Variance)", fontsize=14, labelpad=10, fontstyle='italic')
    ax.set_ylabel("PCA2 (12.06% Variance)", fontsize=14, labelpad=10, fontstyle='italic')

    # Настройка сетки
    ax.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.7, color='gray')

    # Легенда с пояснениями
    legend = ax.legend(
        title="Cluster",
        title_fontsize=12,
        loc="upper right",
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )
    legend.get_frame().set_alpha(0.9)  # Добавляем легкую прозрачность к фону легенды

    # Подпись точек с наиболее важными значениями
    highlight_points = filtered_df[filtered_df['PCA1'].abs() > 30]
    for _, row in highlight_points.iterrows():
        ax.text(
            row['PCA1'], row['PCA2'],
            f"({row['PCA1']:.1f}, {row['PCA2']:.1f})",
            fontsize=9,
            ha='center',
            color='black'
        )

    # Добавление линии нуля для ориентира
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')

    # Улучшение отображения
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    The **Enhanced Cluster Analysis of Financial Data** graph provides a visual representation of clusters formed based on the two most significant principal components (PCA1 and PCA2). It helps identify patterns or groupings among financial data points and offers insights into their relationships.

    #### Key Elements:
    1. **Principal Components:**
       - **PCA1**: Shown on the X-axis, it captures 24.30% of the data variance and highlights the most significant financial patterns.
       - **PCA2**: Displayed on the Y-axis, it captures 12.06% of the variance and complements PCA1 by showing orthogonal variations.

    2. **Clusters:**
       - Data points are grouped into distinct clusters (e.g., 0, 1, 2) based on their financial characteristics.
       - Each cluster is represented by a unique color and marker style, aiding in distinguishing groups.

    3. **Highlighted Points:**
       - Key outliers or points with extreme PCA1 values (e.g., beyond ±30) are labeled for further investigation.

    #### Purpose:
    - To reveal **hidden groupings or relationships** within the financial data.
    - To provide a foundation for **predictive models**, helping analysts understand patterns that may indicate bankruptcy risks.
    - To assist in **decision-making**, enabling stakeholders to target specific clusters for intervention or support.

    This visualization is particularly valuable for analyzing complex datasets by reducing dimensionality while retaining meaningful insights. It can guide strategies for risk management, investment decisions, and financial health assessments.
    """)

if bankrupt_button == 'Bankrupt':

    st.markdown('### Let`s look on what the bankrupts indicators show.')

    st.dataframe(filtered_df, use_container_width=True)

    filtered_df_numeric = filtered_df.drop(columns=['Bankrupt?'], errors='ignore')
    filtered_df_numeric = filtered_df_numeric.drop_duplicates()
    filtered_df_numeric = filtered_df_numeric.select_dtypes(include=['float', 'int'])
    corr_matrix = filtered_df_numeric.corr()
    high_corr_matrix = corr_matrix[corr_matrix > 0.9].dropna(how='all', axis=0).dropna(how='all', axis=1)

    import networkx as nx
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    for i in high_corr_matrix.index:
        for j in high_corr_matrix.columns:
            if i != j and abs(high_corr_matrix.loc[i, j]) > 0.7:
                graph.add_edge(i, j, weight=high_corr_matrix.loc[i, j])

    # Расчет степени связей для узлов
    node_sizes = [1000 * graph.degree(node) for node in graph.nodes]

    # Цвет узлов на основе степени связи
    node_colors = [graph.degree(node) for node in graph.nodes]

    # Позиционирование узлов
    pos = nx.spring_layout(graph, seed=150, k=3)

    # Создание графика
    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,  # Цветовая палитра
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=[2 * graph[u][v]['weight'] for u, v in graph.edges],  # Толщина ребер
    )

    # Добавление весов на связи
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={k: f'{v:.2f}' for k, v in labels.items()},
        font_size=8,
        label_pos=0.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Настройка заголовка
    plt.title("Correlation Graph of High Correlations", fontsize=16, pad=20, color="#333333")
    plt.tight_layout()
    st.subheader('Key Insights from the Correlation Graph')
    st.pyplot(fig)

    st.markdown("""
    The Correlation Graph visualizes the relationships between highly correlated financial indicators, highlighting the following key observations:

    1. **Highly Interconnected Variables:**
       - The graph demonstrates clusters of variables with strong interrelationships (correlation > 0.9). These clusters indicate groups of features that share similar trends or dependencies.

    2. **Central Variables:**
       - The nodes with the largest size represent variables with the most connections to others. These central variables play a critical role in understanding the financial dynamics of the dataset.

    3. **Edge Weights:**
       - The thickness of the lines (edges) indicates the strength of the correlation between two variables. Thicker lines signify stronger relationships.

    4. **Cluster Interpretation:**
       - Clusters in the graph may represent underlying patterns, such as financial ratios that are closely linked due to their mathematical or business relationships.

    5. **Insights for Feature Selection:**
       - This visualization can guide feature selection by identifying redundant variables. Strongly correlated variables may carry overlapping information, which could be streamlined in predictive modeling.

    This graph is particularly useful for analyzing the structural relationships in financial data and uncovering critical features for further investigation.
    """)

    # Данные для визуализации
    columns = [
        'ROA(C) before interest and depreciation before interest',
        'Current Liability to Assets',
        'Debt ratio %',
        'Net Value Per Share (A)',
        'Cash flow rate',
        'Realized Sales Gross Margin',
        'Borrowing dependency'
    ]

    # Отфильтровать данные только для банкротов
    bankrupt_data = filtered_df[filtered_df['Bankrupt?'] == 1][columns]

    # Нормализация данных (от 0 до 1)
    normalized_data = (bankrupt_data - bankrupt_data.min()) / (bankrupt_data.max() - bankrupt_data.min())
    mean_values = normalized_data.mean()

    # Построение Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Добавляем оси
    plt.xticks(angles[:-1], columns, fontsize=10)
    ax.set_rscale('linear')

    # Графики
    values = mean_values.tolist()
    values += values[:1]  # Замыкаем круг
    ax.plot(angles, values, label='Average (Bankrupt)', linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.3)

    # Подписываем и добавляем легенду
    plt.title('Radar Chart: Key Metrics for Bankrupt Companies', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.tight_layout()
    st.subheader('Key Insights from the Radar Chart')
    st.pyplot(fig)

    st.markdown('''
                
    
    The **Radar Chart** provides an overview of key financial metrics for bankrupt companies, allowing for a clear comparison across critical areas:
    
    1. **Profitability:**
       - **ROA(A) before interest and depreciation before tax**: Reflects the company`s ability to generate profits relative to its assets, a critical metric for understanding financial stability.
    
    2. **Liquidity:**
       - **Current Liability to Assets**: Highlights short-term financial obligations relative to assets, with higher values indicating potential liquidity challenges common in bankrupt firms.
    
    3. **Debt Dependency:**
       - **Debt ratio** and **Borrowing dependency**: Demonstrate reliance on debt financing, which can exacerbate financial distress and increase bankruptcy risk.
    
    4. **Market Value:**
       - **Net Value Per Share (A)**: Represents market valuation, often lower for bankrupt companies due to reduced investor confidence and financial instability.
    
    5. **Cash Flow Management:**
       - **Cash Flow rate**: Indicates the efficiency of generating cash from operations, vital for maintaining operations during financial crises.
    
    6. **Operational Efficiency:**
       - **Realized Sales Gross Margin**: Measures profitability from core business operations, shedding light on operational performance during financial stress.
    
    This visualization helps identify areas where bankrupt companies exhibit distinct weaknesses, offering a foundation for deeper financial analysis and strategy development.
    ''')



    # Фильтрация данных только для банкротов
    bankrupt_data = filtered_df[filtered_df['Bankrupt?'] == 1]

    # Создание гистограммы с улучшенными параметрами
    fig, ax = plt.subplots(figsize=(12, 8))  # Увеличенный размер графика
    sns.histplot(
        data=bankrupt_data,
        x='Net Value Per Share (A)',  # Ключевая метрика
        bins=25,  # Более точная разбивка на интервалы
        kde=True,  # Добавление линии плотности
        color='#E63946',  # Красивый оттенок красного
        edgecolor='black',  # Контур для столбцов
        alpha=0.9  # Умеренная прозрачность для наглядности
    )

    # Настройка заголовка и осей
    ax.set_title(
        'Distribution of Net Value Per Share (A) for Bankrupt Companies',
        fontsize=18, fontweight='bold', pad=20, color='#1D3557'  # Цвет и акцент заголовка
    )
    ax.set_xlabel(
        'Net Value Per Share (A)',
        fontsize=14, labelpad=10, color='#1D3557'  # Цвет и отступ для оси X
    )
    ax.set_ylabel(
        'Frequency',
        fontsize=14, labelpad=10, color='#1D3557'  # Цвет и отступ для оси Y
    )

    # Настройка сетки
    ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

    # Добавление аннотации максимального значения
    max_value = bankrupt_data['Net Value Per Share (A)'].max()
    max_freq = bankrupt_data['Net Value Per Share (A)'].value_counts().max()
    ax.annotate(
        f'Max Value: {max_value:.2f}',
        xy=(max_value, max_freq), xytext=(max_value + 0.02, max_freq + 5),
        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1.5),
        fontsize=12, color='black', weight='bold'
    )

    # Улучшение внешнего вида
    plt.tight_layout()  # Автоматическое улучшение компоновки
    st.subheader('Key Insights by Histogram method')
    st.pyplot(fig)

    st.markdown('''

    The histogram illustrates the distribution of **Net Value Per Share (A)** for non-bankrupt companies. This metric provides a detailed view of the financial health and operational efficiency of the companies that have managed to avoid financial distress.
    
    #### Observations:
    1. **Concentration Zone**:
       - The data is highly concentrated around a specific range of values, indicating that most non-bankrupt companies maintain a consistent and stable net value per share.
       - The peak frequency suggests the most common net value range that non-bankrupt companies achieve.
    
    2. **Range of Distribution**:
       - The distribution covers a range from lower to higher values of **Net Value Per Share (A)**, reflecting the diversity in the financial strategies and performance among these companies.
    
    3. **Max Value**:
       - The maximum value observed on the histogram highlights the upper threshold achieved by the best-performing companies in terms of net value per share.
    
    4. **Stability**:
       - The lack of extreme outliers in the distribution suggests a relatively stable financial performance across the non-bankrupt group.
    
    #### Interpretation:
    This histogram helps analysts understand the financial characteristics of non-bankrupt companies, emphasizing their ability to maintain positive net values. Such stability can serve as a benchmark for evaluating companies on the brink of financial instability. Furthermore, the concentration zones highlight typical financial performance ranges that are common for successful companies.
    
    By comparing this distribution to that of bankrupt companies, stakeholders can identify key differences and potential thresholds critical for financial decision-making.
    ''')



    # Фильтрация данных только для банкротов
    bankrupt_data = filtered_df[filtered_df['Bankrupt?'] == 1]

    # Создание scatterplot
    fig, ax = plt.subplots(figsize=(12, 8))  # Увеличенный размер графика
    sns.scatterplot(
        data=bankrupt_data,
        x='Debt ratio %',  # Ось X
        y='Current Liability to Assets',  # Ось Y
        size='Cash flow rate',  # Размер точки
        hue='ROA(C) before interest and depreciation before interest',  # Цветовая шкала по значению
        palette='coolwarm',  # Цветовая палитра
        sizes=(50, 300),  # Диапазон размеров точек
        alpha=0.8,  # Прозрачность
        edgecolor='black',  # Цвет контура точек
        linewidth=0.5,  # Толщина контура точек
        ax=ax  # Указание оси
    )

    # Настройка заголовка и осей
    ax.set_title(
        'Relationship Between Debt ratio %, Current Liability to Assets, and ROA(C)',
        fontsize=18, fontweight='bold', pad=20, color='#1D3557'
    )
    ax.set_xlabel(
        'Debt ratio %',
        fontsize=14, labelpad=10, color='#1D3557'
    )
    ax.set_ylabel(
        'Current Liability to Assets',
        fontsize=14, labelpad=10, color='#1D3557'
    )

    # Настройка сетки
    ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

    # Добавление аннотации максимального значения
    max_point = bankrupt_data.loc[bankrupt_data['ROA(C) before interest and depreciation before interest'].idxmax()]
    ax.annotate(
        f'Max: {max_point["ROA(C) before interest and depreciation before interest"]:.2f}',
        xy=(max_point['Debt ratio %'], max_point['Current Liability to Assets']),
        xytext=(
            max_point['Debt ratio %'] + 0.05,
            max_point['Current Liability to Assets'] + 0.02
        ),
        arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1.5),
        fontsize=12, color='black', weight='bold'
    )

    # Улучшение внешнего вида
    plt.tight_layout()  # Автоматическое улучшение компоновки
    st.subheader('Now let`s look on on the behaviour of some particular metrics by Scatter Plot')
    st.pyplot(fig)

    st.markdown('''
    This scatterplot provides valuable insights into the financial characteristics of bankrupt companies by examining the relationship between the debt ratio, current liability to assets, and ROA(C) before interest and depreciation:
    
    1. **Clusters and Patterns**:
       - Companies with higher debt ratios often exhibit higher current liabilities to assets, indicating potential liquidity risks.
       - A distinct clustering is observed where companies with lower ROA(C) values have significantly higher debt burdens, potentially highlighting financial inefficiencies.
    
    2. **Size and Color Analysis**:
       - The size of the points, determined by the cash flow rate, varies across the chart. Companies with lower cash flow rates are typically represented by smaller points, further emphasizing their inability to manage operational liquidity effectively.
       - The color gradient reflects ROA(C) values, where darker hues indicate better profitability. Most bankrupt companies are concentrated in the lighter-colored areas, signifying lower profitability and financial stress.
    
    3. **Key Insights**:
       - The chart highlights the financial stress zones for bankrupt companies, where high debt ratios coincide with low profitability.
       - Outliers with unusually high ROA(C) may represent rare cases of temporary financial recovery or unique financial strategies that require deeper investigation.
    
    ### Conclusion:
    This visualization underscores the importance of maintaining a balance between debt obligations and asset efficiency. It serves as a critical tool for identifying financial distress patterns in bankrupt companies, enabling analysts to pinpoint factors contributing to financial instability.
    ''')



    # Создание улучшенного pairplot
    pairplot = sns.pairplot(
        data=bankrupt_data,
        vars=['Current Liability to Assets', 'Net Value Per Share (A)', 'Cash flow rate'],
        palette='viridis',  # Использование последовательной палитры
        diag_kind='kde',  # Диагональные элементы: KDE
        markers=['o', 's'],  # Формы точек
        plot_kws={
            'alpha': 0.8,  # Прозрачность точек
            's': 70,  # Размер точек
            'edgecolor': 'gray',  # Цвет границы точек
            'linewidth': 0.5  # Толщина границы
        },
        diag_kws={
            'shade': True,  # Подсветка под KDE
            'alpha': 0.7  # Прозрачность
        }
    )

    # Настройка заголовка
    pairplot.fig.suptitle(
        "Pairplot Analysis of Selected Financial Metrics for Bankrupt Companies",
        fontsize=22,
        fontweight='bold',
        color='#3B3B3B',
        y=1.05  # Смещение заголовка
    )

    # Настройка легенды
    for ax in pairplot.axes.flatten():
        if ax:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Настройка сетки
        if ax.get_legend():
            ax.legend(
                title="Hue Legend",
                fontsize=12,
                title_fontsize=14,
                loc='upper right',
                frameon=True
            )

    # Показываем график
    st.subheader('Key Insights from the Pairplot Analysis:')
    st.pyplot(pairplot)

    st.markdown("""
    

    1. **Interrelationships Between Metrics**:
       - The pairplot visualizes the relationships between key financial metrics, such as **Current Liability to Assets**, **Net Value Per Share (A)**, and **Cash Flow Rate**.
       - Observing the scatter patterns can reveal correlations or lack thereof, helping to identify key dependencies between metrics.

    2. **Distribution of Metrics**:
       - The diagonal KDE plots provide a clear view of the individual distributions for each metric. For example:
         - **Current Liability to Assets** shows a concentrated peak, indicating a common range for most bankrupt companies.
         - **Net Value Per Share (A)** highlights variations in equity performance across companies.
         - **Cash Flow Rate** demonstrates clustering near lower values, reflecting potential liquidity challenges.

    3. **Cluster Identification**:
       - By analyzing the scatter points, you can observe clusters or outliers that may indicate distinct financial behaviors or anomalies.

    4. **Visual Patterns**:
       - The pairplot emphasizes financial stress zones for bankrupt companies, such as regions where liabilities dominate or cash flow rates decline sharply.

    This visualization helps analysts better understand the financial dynamics of bankrupt companies and provides a foundation for identifying potential risk factors and areas for improvement.
    """)

if bankrupt_button == 'Non-Bankrupt':
    st.markdown('### And now let`s look on indicators which belong to non-bankrupt companies.')

    st.dataframe(filtered_df, use_container_width=True)

    filtered_df_numeric = filtered_df.drop(columns=['Bankrupt?'], errors='ignore')
    filtered_df_numeric = filtered_df_numeric.drop_duplicates()
    filtered_df_numeric = filtered_df_numeric.select_dtypes(include=['float', 'int'])
    corr_matrix = filtered_df_numeric.corr()
    high_corr_matrix = corr_matrix[corr_matrix > 0.9].dropna(how='all', axis=0).dropna(how='all', axis=1)

    import networkx as nx
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    for i in high_corr_matrix.index:
        for j in high_corr_matrix.columns:
            if i != j and abs(high_corr_matrix.loc[i, j]) > 0.7:
                graph.add_edge(i, j, weight=high_corr_matrix.loc[i, j])

    # Расчет степени связей для узлов
    node_sizes = [1000 * graph.degree(node) for node in graph.nodes]

    # Цвет узлов на основе степени связи
    node_colors = [graph.degree(node) for node in graph.nodes]

    # Позиционирование узлов
    pos = nx.spring_layout(graph, seed=150, k=3)

    # Создание графика
    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,  # Цветовая палитра
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=[2 * graph[u][v]['weight'] for u, v in graph.edges],  # Толщина ребер
    )

    # Добавление весов на связи
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={k: f'{v:.2f}' for k, v in labels.items()},
        font_size=8,
        label_pos=0.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    # Настройка заголовка
    plt.title("Correlation Graph of High Correlations", fontsize=16, pad=20, color="#333333")
    plt.tight_layout()
    st.subheader('Correlation Graph: Identifying Patterns in Non-Bankrupt Companies')
    st.pyplot(fig)
    st.markdown("""
    

    As we shift our focus to analyzing the financial indicators of **non-bankrupt companies**, this correlation graph serves as a foundational tool to uncover the relationships between key metrics. By visualizing the strength and direction of correlations, we aim to identify significant patterns that can differentiate the performance and stability of non-bankrupt companies from their bankrupt counterparts.

    #### Key Highlights:
    1. **High Correlation Nodes**:
       - Metrics such as **Net Value Per Share (C)**, **Cash Flow Rate**, and **Operating Gross Margin** show strong interdependencies, suggesting their critical role in the financial performance of non-bankrupt companies.

    2. **Clusters of Relationships**:
       - The graph reveals clusters where metrics are closely interconnected, which may indicate shared influences or dependencies. For instance, **Liability to Equity** and **Realized Sales Gross Margin** could reflect a company's ability to balance debt and profitability.

    3. **Actionable Insights**:
       - High positive correlations between certain metrics (thicker edges) might signal key drivers of financial stability, while weaker or negative correlations could point to areas of risk or inefficiency.

    This correlation-based approach provides a strategic starting point for understanding the resilience and success factors of non-bankrupt companies, paving the way for more targeted analyses.
    """)

    columns = [
        'Operating Gross Margin',
        'Net Value Per Share (C)',
        'Cash flow rate',
        'Liability to Equity',
        'Realized Sales Gross Margin',
        'After-tax Net Profit Growth Rate',
        'Regular Net Profit Growth Rate'
    ]

    # Выбор колонок для анализа корреляции
    columns_to_plot = [
        'Operating Gross Margin',
        'Net Value Per Share (C)',
        'Cash flow rate',
        'Liability to Equity',
        'Realized Sales Gross Margin',
        'After-tax Net Profit Growth Rate',
        'Regular Net Profit Growth Rate'
    ]

    # Фильтрация данных только для небанкротов
    non_bankrupt_data = filtered_df[filtered_df['Bankrupt?'] == 0]

    # Вычисление корреляции
    correlation_matrix = non_bankrupt_data[columns_to_plot].corr()

    # Создание булевой маски для скрытия значений выше диагонали
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # k=1 оставляет диагональ видимой

    # Улучшенная версия тепловой карты
    plt.figure(figsize=(12, 10))  # Увеличенный размер графика для лучшей читаемости
    sns.heatmap(
        correlation_matrix,
        mask=mask,  # Применение маски
        annot=True,  # Отображение значений
        cmap="coolwarm",  # Цветовая палитра
        fmt=".2f",  # Формат значений (2 знака после запятой)
        annot_kws={"size": 10},  # Настройка размера шрифта аннотаций
        linewidths=1,  # Более толстые линии между ячейками
        linecolor="gray",  # Цвет линий между ячейками
        cbar_kws={
            "shrink": 0.8,  # Уменьшение цветовой шкалы
            "aspect": 20,  # Соотношение размеров цветовой шкалы
            "pad": 0.05,  # Расстояние между цветовой шкалой и тепловой картой
        }
    )

    # Настройка заголовка
    plt.title(
        "Enhanced Correlation Heatmap of Financial Metrics for Non-Bankrupt Companies",
        fontsize=20,
        fontweight="bold",
        pad=30,
        color="#333333"  # Цвет заголовка
    )

    # Настройка шрифтов осей
    plt.xticks(fontsize=12, rotation=45, ha="right", color="#333333")  # Поворот текста на осях X
    plt.yticks(fontsize=12, rotation=0, color="#333333")  # Текст на оси Y без поворота

    # Улучшение отображения
    plt.tight_layout()  # Автоматическое улучшение расположения элементов
    st.subheader('Insights from the Correlation Heatmap')
    st.pyplot(plt)

    # Генерация st.markdown для пользователя
    st.markdown("""
    
    
    This heatmap visualizes the correlations between selected financial metrics of **non-bankrupt companies**. Here are some key takeaways:

    1. **Strong Positive Correlations:**
       - Metrics such as **Liability to Equity** and **Cash Flow rate** show a strong positive correlation, indicating potential relationships between financial stability and operational efficiency.

    2. **Weak or Negative Correlations:**
       - **Net Value Per Share (C)** has weaker correlations with other metrics, suggesting that its variations may be driven by different financial factors.

    3. **Clusters of Metrics:**
       - Metrics such as **Operating Gross Margin** and **Realized Sales Gross Margin** form a cluster of related metrics, emphasizing profitability and operational performance.

    """)

    # Настройка темы графика
    sns.set_theme(style="whitegrid")

    # Создание гистограммы
    fig, ax = plt.subplots(figsize=(10, 8))  # Увеличенный размер графика
    sns.histplot(
        data=filtered_df[filtered_df['Bankrupt?'] == 0],  # Используем данные только для небанкротов
        x='Net Value Per Share (C)',  # Метрика для анализа
        bins=25,  # Количество столбцов
        kde=True,  # Линия плотности
        color="#4A90E2",  # Цвет столбцов
        edgecolor="black",  # Цвет границ столбцов
        alpha=0.8  # Прозрачность
    )

    # Настройка заголовка и подписей осей
    ax.set_title(
        "Distribution of Net Value Per Share (C) for Non-Bankrupt Companies",
        fontsize=18,
        fontweight="bold",
        pad=20,
        color="#333333"
    )
    ax.set_xlabel(
        "Net Value Per Share (C)",
        fontsize=14,
        labelpad=10,
        color="#333333"
    )
    ax.set_ylabel(
        "Frequency",
        fontsize=14,
        labelpad=10,
        color="#333333"
    )

    # Настройка сетки
    ax.grid(visible=True, linestyle="--", linewidth=0.7, alpha=0.6, color="gray")

    # Финализация графика
    plt.tight_layout()
    st.subheader('Insights from the Distribution of Net Value Per Share (C) for Non-Bankrupt Companies')
    st.pyplot(fig)
    st.markdown("""
    The histogram and KDE visualization provide a comprehensive view of the distribution of the **Net Value Per Share (C)** for non-bankrupt companies. Here are the key takeaways:

    #### Key Observations:
    1. **Concentration Zone:**
       - The majority of non-bankrupt companies have a **Net Value Per Share (C)** within a specific range, with the peak density observed near **0.2**.
       - This indicates that most non-bankrupt companies are maintaining relatively stable net values.

    2. **Distribution Shape:**
       - The distribution has a sharp peak with a gradual tail, suggesting a small number of companies exhibit higher values. 

    3. **Financial Health:**
       - The absence of extreme outliers emphasizes the financial stability and consistency among non-bankrupt companies.

    
    By examining this metric, we can better understand the financial characteristics that distinguish non-bankrupt companies and identify patterns contributing to their success.
    """)

    # Создание scatterplot для анализа двух метрик
    fig, ax = plt.subplots(figsize=(12, 8))  # Увеличенный размер графика
    sns.scatterplot(
        data=filtered_df[filtered_df['Bankrupt?'] == 0],  # Используем данные только для небанкротов
        x='Current Liability to Assets',  # Ось X
        y='Cash flow rate',  # Ось Y
        size='Net Value Per Share (C)',  # Размер точки
        hue='Realized Sales Gross Margin',  # Цветовая шкала по значению
        palette='coolwarm',  # Цветовая палитра
        sizes=(50, 300),  # Диапазон размеров точек
        alpha=0.8,  # Прозрачность
        edgecolor='black',  # Цвет границы точек
        linewidth=0.5,  # Толщина границы точек
        ax=ax  # Указание оси
    )

    # Настройка заголовка и подписей осей
    ax.set_title(
        'Relationship Between Current Liability to Assets and Cash Flow Rate (Non-Bankrupt)',
        fontsize=18, fontweight='bold', pad=20, color='#1D3557'
    )
    ax.set_xlabel(
        'Current Liability to Assets',
        fontsize=14, labelpad=10, color='#1D3557'
    )
    ax.set_ylabel(
        'Cash Flow Rate',
        fontsize=14, labelpad=10, color='#1D3557'
    )

    # Настройка сетки
    ax.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.6, color='gray')

    # Добавление легенды
    ax.legend(title="Realized Sales Gross Margin", fontsize=12, title_fontsize=14, loc='upper right')

    # Финализация графика
    plt.tight_layout()
    st.subheader('Insights from the Relationship Between Current Liability to Assets and Cash Flow Rate')
    st.pyplot(fig)

    # st.markdown для пользователя
    st.markdown("""
    This scatterplot highlights the interplay between **Current Liability to Assets** and **Cash Flow Rate** for non-bankrupt companies. The visualization offers the following insights:

    #### Key Observations:
    1. **Clusters of Companies:**
       - Companies with low **Current Liability to Assets** often show higher **Cash Flow Rates**, indicating better financial liquidity and operational efficiency.
       - A few clusters appear where both metrics are balanced, representing financially stable companies.

    2. **Impact of Net Value Per Share (C):**
       - The size of the points corresponds to **Net Value Per Share (C)**, where larger points indicate companies with higher net values. These are generally clustered in regions of financial stability.

    3. **Realized Sales Gross Margin:**
       - The color gradient shows variations in **Realized Sales Gross Margin**, offering insights into how profitability impacts operational efficiency.

    """)

     # Настройка темы графика
    sns.set_theme(style="darkgrid")

     # Выбор метрик для анализа
    metrics_to_plot = ['Liability to Equity', 'Realized Sales Gross Margin']
    #
    # # Создание графиков KDE
    plt.figure(figsize=(12, 8))  # Увеличенный размер графика
    for metric in metrics_to_plot:
         sns.kdeplot(
             data=filtered_df[filtered_df['Bankrupt?'] == 0],  # Данные только для небанкротов
             x=metric,  # Текущая метрика
             fill=True,  # Заливка под графиком
             alpha=0.7,  # Прозрачность
             linewidth=2.5,  # Толщина линии
             label=metric,  # Легенда для текущей метрики
         )
    #
    # # Настройка заголовка
    plt.title(
         'KDE Analysis: Liability to Equity vs Realized Sales Gross Margin (Non-Bankrupt)',
         fontsize=18,
         fontweight='bold',
         pad=20,
         color="#333333"
     )

     # Настройка осей
    plt.xlabel("Value", fontsize=14, labelpad=10, color="#333333")
    plt.ylabel("Density", fontsize=14, labelpad=10, color="#333333")
    #
    # # Настройка легенды
    plt.legend(
         title="Metrics",
         fontsize=12,
         title_fontsize=14,
         loc="upper right",
         frameon=True,
         shadow=True,
         fancybox=True,
         borderpad=1.5
     )
    #
    # # Настройка сетки
    plt.grid(visible=True, linestyle='--', linewidth=0.7, alpha=0.6, color="gray")
    #
    # # Финализация графика
    plt.tight_layout()
    st.subheader('Insights from the KDE Analysis: Liability to Equity vs Realized Sales Gross Margin')
    st.pyplot(plt)
    #
    # # Генерация st.markdown для анализа
    st.markdown("""
    
     This KDE analysis provides a detailed comparison of **Liability to Equity** and **Realized Sales Gross Margin** metrics for non-bankrupt companies.
    
     ### Key Observations:
     1. **High Concentration Zones:**
        - Both metrics exhibit distinct density peaks, indicating zones where most companies fall.
        - The peak for **Liability to Equity** suggests efficient equity management among non-bankrupt companies.
    
     2. **Variability:**
        - **Realized Sales Gross Margin** shows wider variability, reflecting differences in operational efficiency across companies.
    
     """)

    # Пример фильтрации данных только для небанкротов
    non_bankrupt_data = filtered_df[filtered_df['Bankrupt?'] == 0]

    # Колонки для анализа
    columns_to_analyze = ['Operating Gross Margin', 'Liability to Equity', 'Realized Sales Gross Margin']

    # Вычисление среднего значения и стандартного отклонения
    mean_std_data = non_bankrupt_data[columns_to_analyze].agg(['mean', 'std']).T
    mean_std_data.columns = ['mean', 'std']  # Переименование колонок для удобства
    mean_std_data['Metric'] = mean_std_data.index

    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 7))  # Увеличенный размер графика
    sns.barplot(
        data=mean_std_data,
        x='Metric',
        y='mean',
        palette='coolwarm',  # Цветовая палитра для улучшенной визуализации
        edgecolor='black',  # Черные края для улучшенной четкости
        linewidth=1.5,  # Толщина границ баров
        ax=ax
    )

    # Добавление погрешностей вручную
    for i, row in enumerate(mean_std_data.itertuples()):
        ax.errorbar(
            x=i,
            y=row.mean,
            yerr=row.std,
            fmt='o',  # Метка ошибки
            c='darkred',  # Цвет погрешностей
            capsize=5,  # Размер "шляпок" на погрешностях
            alpha=0.8,  # Прозрачность
            capthick=1.5  # Толщина "шляпок"
        )

    # Настройка заголовка и осей
    ax.set_title('Enhanced Bar Chart with Error Bars for Non-Bankrupt Companies', fontsize=18, fontweight='bold',
                 pad=15, color='#4A4A4A')
    ax.set_xlabel('Metrics', fontsize=14, labelpad=10, color='#4A4A4A')
    ax.set_ylabel('Mean Value with Standard Deviation', fontsize=14, labelpad=10, color='#4A4A4A')

    # Добавление аннотаций для среднего значения
    for i, row in enumerate(mean_std_data.itertuples()):
        ax.text(
            x=i,
            y=row.mean + row.std + 0.02,  # Немного выше погрешности
            s=f'{row.mean:.2f}',
            ha='center',
            fontsize=12,
            fontweight='bold',
            color='darkblue'
        )

    # Улучшение внешнего вида
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')
    plt.xticks(rotation=30, ha='right', fontsize=12, color='#4A4A4A')  # Поворот подписей с выравниванием
    plt.yticks(fontsize=12, color='#4A4A4A')

    # Финализация графика
    plt.tight_layout()
    st.subheader('Insights from the Bar Chart with Error Bars')
    st.pyplot(fig)

    # Markdown для пояснений
    st.markdown("""
    

    This bar chart illustrates the **average values** of key financial metrics for non-bankrupt companies, with **error bars** representing the standard deviation. 

    #### Key Takeaways:
    1. **Metrics Comparison**: The chart allows for a quick comparison of average values across metrics such as:
       - Operating Gross Margin
       - Liability to Equity
       - Realized Sales Gross Margin
    2. **Variability**: The size of the error bars highlights the variability within each metric:
       - Larger error bars indicate more variability in the data.
       - Smaller error bars suggest more consistency across the group.
    3. **Business Insights**:
       - Metrics with low variability could indicate stable financial performance.
       - High variability may suggest potential risks or areas for improvement.

    This visualization provides a foundation for deeper analysis of financial stability and operational performance among non-bankrupt companies.
    """)







