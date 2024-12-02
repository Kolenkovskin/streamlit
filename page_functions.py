import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px



df = pd.read_csv('./data.csv')
df.columns = df.columns.str.strip()
df = df.drop(['Net Income Flag', 'Liability-Assets Flag'], axis=1)


def col_analyse(col):

    import streamlit as st
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore, levene, ttest_ind, mannwhitneyu
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Блок 1: Вычисление основных характеристик
    # Фильтрация данных по условию
    st.subheader(f'The {col}.')
    st.write(f'Here you can estimate a column {col} separately!')
    st.dataframe(df[col])

    data_sum = df[col].sum()

    if data_sum >= 10000:
        col_data = df[df[col] > 1][col]  # Фильтруем строки, где значения > 1
        st.subheader('Main statistical view.')
        st.write(f"The column below provides quantitative data of {col} that describes a specific financial metric with main indicators shown.")
        st.write(col_data.describe())
        st.write()
    elif (df[col] == 1).any() or (df[col] == 0).any():
        col_data = df[col][(df[col] > 0) & (df[col] < 1)]
        st.subheader('Main statistical view.')
        st.write(f"The column below provides quantitative data of {col} that describes a specific financial metric with main indicators shown.")
        st.write(col_data.describe())
        st.write()
    else:
        col_data = df[col]  # Без фильтрации
        st.subheader('Main statistical view.')
        st.write(f"The column below provides quantitative data of {col} that describes a specific financial metric.")
        st.write(col_data.describe())
        st.write()


    # Вычисление выбросов
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    col_data = col_data.dropna()
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

    if len(outliers) > 0:
        st.markdown(f'### The {col} column has {len(outliers)} of interquartile outliers.')
        st.write()
        st.write(outliers)
        st.write()
        st.markdown("""
                ### Interquartile Outliers

                Interquartile outliers refer to data points that fall significantly outside the range of most of the data within a dataset. These outliers are identified using the **interquartile range (IQR)**, which is the difference between the **75th percentile (Q3)** and the **25th percentile (Q1)** of the data. The IQR represents the range where the middle 50% of the data lies.

                ### General Concept:
                 - A data point is considered an interquartile outlier if it falls:
                   - **Below Q1 - 1.5 × IQR**
                   - **Above Q3 + 1.5 × IQR**

                ### Significance:
                 1. **Outliers Detection**: Interquartile outliers highlight values that are unusually high or low compared to the majority of the data.
                 2. **Data Quality**: These outliers may indicate errors, rare events, or extreme variations in the data.
                 3. **Impact on Analysis**: Outliers can significantly influence statistical measures like the mean and standard deviation, so identifying them is crucial for robust analysis.
                 4. **Actionable Insights**: In financial or business contexts, interquartile outliers might signal abnormal behavior, potential risks, or opportunities for further investigation.

                 Interquartile outliers provide a structured way to detect and interpret deviations in data, which helps in improving the reliability of statistical analyses and decision-making processes.
                 """)
        st.markdown(f"""
                         ### Boxplot of {col} outliers.
                         This boxplot visualizes the statistical summary of the column **{col}**, including its median, quartiles, and outliers.
                         The box represents the interquartile range (IQR), while the whiskers show the range within 1.5 times the IQR.
                         Any points outside this range are considered outliers, offering insights into potential anomalies in the data.
                         """)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=col_data, ax=ax, color="skyblue")
        ax.set_title(f'Enhanced Boxplot of {col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{col}', fontsize=12)
        ax.set_ylabel('Value Distribution', fontsize=12)
        ax.grid(visible=True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        st.write()

    # Вычисление Z-выбросов
    z_scores = zscore(col_data)
    z_outliers = col_data[np.abs(z_scores) > 3]
    if len(z_outliers) > 0:
        st.markdown(f'### Also the {col} column has {len(z_outliers)} of Z-score outliers.')
        st.write()
        st.write(z_outliers)
        st.write()
        st.markdown("""
            ### Z-Score Outliers

            **Z-score outliers** are data points that deviate significantly from the mean of a dataset when measured in terms of standard deviations. The Z-score measures how many standard deviations a data point is from the mean, providing a standardized way to identify extreme values.

            #### Significance:
            1. **Outliers Detection**: Z-scores help identify both high and low extreme values in a dataset.
            2. **Standardized Scale**: Unlike raw values, Z-scores are unitless and allow for comparisons across different datasets.
            3. **Data Analysis**: Recognizing Z-score outliers is critical for understanding data variability, removing noise, and ensuring robust statistical results.
            4. **Applications**: In areas such as finance, healthcare, or engineering, Z-score outliers might indicate anomalies, trends, or rare events worth further analysis.

            By using Z-scores, analysts can systematically identify and evaluate extreme data points, ensuring reliable and actionable insights.
            """)
        st.markdown(f"""
            ### Z-Score Scatter Plot for {col}

            This scatter plot visualizes the Z-scores for the selected column **{col}**.
            Each point represents a data observation, with its Z-score calculated based on the column's mean and standard deviation.

            - **Points above the orange line at +3** indicate positive outliers, which are significantly above the average.
            - **Points below the orange line at -3** indicate negative outliers, which are significantly below the average.

            Understanding this distribution helps to identify and analyze extreme values that might impact the data's overall interpretation or model performance.
            """)
        # Настройка графика
        fig, ax = plt.subplots(figsize=(12, 6))  # Увеличенный размер для лучшей визуализации

        # Отображение точек Z-оценок
        ax.scatter(range(len(col_data)), z_scores, alpha=0.7, label="Z-scores", color="blue")

        # Горизонтальные линии порогов
        ax.axhline(y=3, color="red", linestyle="--", linewidth=2, label="Upper Threshold (+3)")
        ax.axhline(y=-3, color="green", linestyle="--", linewidth=2, label="Lower Threshold (-3)")

        # Установление границ осей
        ax.set_xlim(0, len(col_data))  # Ось X охватывает весь диапазон данных
        ax.set_ylim(-5, 5)  # Ось Y охватывает значения от -5 до 5 для лучшего контекста

        # Подписи для выбросов
        # Убедимся, что extreme_points — это Series
        extreme_points = z_outliers[abs(z_outliers) > 4]  # Это фильтрация должна вернуть Series

        # Если `extreme_points` — это Series
        if isinstance(extreme_points, pd.Series):
            for idx, val in extreme_points.items():  # Используем .items() вместо .iteritems()
                ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=8, color='purple')
        else:
            st.write("Extreme points were not identified or are not in the correct format.")

        # Сетка для удобства анализа
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        # Заголовок и подписи осей
        ax.set_title(f"Z-Score Scatter Plot for {col}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Index", fontsize=14)
        ax.set_ylabel("Z-Score", fontsize=14)

        # Легенда
        ax.legend(loc="upper left", fontsize=10)

        # Вывод графика
        st.pyplot(fig)
        st.write(
            "This scatter plot shows the Z-scores of the selected column. Points outside the thresholds (+3/-3) are potential outliers.")

    # Блок 2: Корреляции
    num_df = df.drop(columns=['Bankrupt?'])
    correlations = num_df.corr()[col]
    high_correlations = correlations[correlations > 0.7].dropna()
    high_correlations = high_correlations.drop(col, errors='ignore')
    if not high_correlations.empty:
        st.markdown("""
            ### High Correlations

            **High correlations** refer to strong linear relationships between two variables in a dataset. Correlation measures the extent to which changes in one variable are associated with changes in another.

            #### Importance:
            1. **Feature Selection**: In machine learning, highly correlated features might cause redundancy and multicollinearity, leading to less accurate models.
            2. **Predictive Insights**: High correlations often suggest that one variable may predict another, though correlation does not imply causation.
            3. **Understanding Relationships**: High correlations can uncover key relationships between variables, aiding in better data interpretation and decision-making.

            #### Use Case Example:
            For financial datasets, high correlations might indicate:
            - A strong relationship between revenue and expenses.
            - Dependency between stock prices and market indices.
            - Economic trends influencing multiple indicators.

            Understanding and managing high correlations in data ensures robust analysis and accurate conclusions.
            """)
        st.write()
        st.write(high_correlations)
        st.write()
        st.markdown(f"""
            ### High Correlations with {col}

            This bar chart highlights the variables that have a strong linear relationship with the selected column **{col}**.
            Understanding these correlations is key to feature selection and avoiding redundancy in modeling.
            """)
        # Improved bar chart for high correlations
        fig, ax = plt.subplots(figsize=(10, 7))  # Increased figure size for better readability

        # Plot sorted high correlations as a horizontal bar chart for better variable comparison
        high_correlations.sort_values().plot(
            kind='barh',
            ax=ax,
            color='skyblue',
            edgecolor='black',  # Add border to bars for clarity
            alpha=0.8  # Slightly transparent bars
        )

        # Add titles and labels with adjusted font sizes and weights
        ax.set_title(f"High Correlations with {col}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Correlation Coefficient", fontsize=14, fontweight='medium', labelpad=10)
        ax.set_ylabel("Variables", fontsize=14, fontweight='medium', labelpad=10)

        # Enable grid for easier value interpretation
        ax.grid(axis='x', linestyle='--', linewidth=0.7, alpha=0.7, color='gray')

        # Adjust tick parameters for better readability
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Add values next to the bars
        for i, v in enumerate(high_correlations.sort_values()):
            ax.text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=12, color='black')

        # Display the chart
        st.pyplot(fig)

        # Add a note below the chart for explanation
        st.write(
            "This bar chart highlights the variables that have a strong linear relationship with the selected column. Understanding these correlations helps in feature selection and reducing redundancy in data modeling.")

    bankrupt_group = df.loc[df['Bankrupt?'] == 1, col]
    non_bankrupt_group = df.loc[df['Bankrupt?'] == 0, col]


    f_stat, p_value = levene(bankrupt_group, non_bankrupt_group)
    st.markdown(f"**F-statistic:** {f_stat}, **p-value:** {p_value}.\n\nThe **F-statistic** is used in hypothesis testing to compare the variances of two groups. It helps determine whether there is a significant difference between the variability of the groups being tested. A higher **F-value** suggests greater differences in variability between the groups. In the context of financial indicators, the F-statistic can reveal if certain metrics exhibit greater variability in bankrupt versus non-bankrupt companies, highlighting potentially significant features for analysis.")
    if p_value > 0.05:
        t_stat, p_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        st.markdown(f"**T-statistic:** {t_stat}, **p-value:** {p_value}.\n\nThe **T-statistic** assesses whether the means of two groups differ significantly. It measures the ratio of the difference between the group means to the variability within the groups. For financial analysis, the T-statistic indicates whether a specific indicator has distinct average values for bankrupt and non-bankrupt companies. A higher absolute **T-value** suggests stronger evidence against the null hypothesis, meaning the group means are likely different.")
        if p_value > 0.05:
            st.markdown("""
            ### No Statistically Significant Differences
            There are no statistically significant differences between the **bankrupt** and **non-bankrupt** groups in terms of the chosen indicator.
            This implies that the selected indicator does not effectively distinguish between the two groups.
            """)
        else:
            st.markdown("""
            ### Statistically Significant Differences
            Statistically significant differences exist between the **bankrupt** and **non-bankrupt** groups for the selected indicator.
            This indicates that the chosen metric has a measurable impact or distinction between these two categories.
            """)

    else:
        t_stat, p_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        st.markdown(f"**T-statistic:** {t_stat}, **p-value:** {p_value}.\n\nThe **T-statistic** assesses whether the means of two groups differ significantly. It measures the ratio of the difference between the group means to the variability within the groups. For financial analysis, the T-statistic indicates whether a specific indicator has distinct average values for bankrupt and non-bankrupt companies. A higher absolute **T-value** suggests stronger evidence against the null hypothesis, meaning the group means are likely different.")
        if p_value > 0.05:
            st.markdown("""
            ### No Statistically Significant Differences
            Even with unequal variances, there are no statistically significant differences between the **bankrupt** and **non-bankrupt** groups.
            This suggests that the chosen indicator might not be a good discriminator for the two groups.
            """)
        else:
            u_stat, p_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            st.markdown(f"**U-statistic:** {u_stat}, **p-value:** {p_value}.\n\nThe **U-statistic** is derived from the **Mann-Whitney U test**, a non-parametric test that evaluates whether two groups come from the same distribution. It is particularly useful when data is not normally distributed. The **P-value** indicates the probability of observing the test results under the null hypothesis. A low P-value (commonly < 0.05) suggests that the observed differences between the groups are statistically significant. In bankruptcy analysis, the U-statistic and P-value help determine whether financial indicators significantly differentiate bankrupt from non-bankrupt companies without assuming normality in the data.")
            if p_value > 0.05:
                st.markdown("""
                ### No Statistically Significant Differences (U-Test)
                According to the Mann-Whitney U-test, no statistically significant differences are observed between the **bankrupt** and **non-bankrupt** groups.
                This supports the notion that the selected indicator is not a clear differentiator between the two groups.
                """)
            else:
                st.markdown("""
                ### Statistically Significant Differences (U-Test)
                The Mann-Whitney U-test indicates statistically significant differences between the **bankrupt** and **non-bankrupt** groups.
                This highlights that the selected indicator has meaningful variability or impact between these two categories.
                """)
    st.write()
    st.markdown(f"""
        ### Distribution of {col}
        This histogram represents the distribution of the selected column **{col}**.
        It provides an overview of how the data values are spread across different ranges.
        Analyzing this distribution helps to understand the central tendency, spread, and potential skewness in the data.
        """)
    fig, ax = plt.subplots(figsize=(10, 6))  # Увеличен размер графика
    ax.hist(
        col_data,
        bins=30,  # Количество бинов для более детализированного распределения
        alpha=0.75,  # Умеренная прозрачность для лучшей видимости
        color='skyblue',  # Приятный цвет для гистограммы
        edgecolor='black',  # Черные края для четкости бинов
        linewidth=1.2  # Толщина линии края
    )

    # Добавление линий среднего и медианы для анализа распределения
    mean_value = col_data.mean()
    median_value = col_data.median()
    ax.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
    ax.axvline(median_value, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_value:.2f}')

    # Настройка заголовков и меток
    ax.set_title(f"Distribution of {col}", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(col, fontsize=14, labelpad=10)
    ax.set_ylabel('Frequency', fontsize=14, labelpad=10)
    ax.legend(fontsize=12)  # Легенда для пояснения линий

    # Добавление сетки для упрощения интерпретации
    ax.grid(visible=True, linestyle='--', alpha=0.6)

    st.pyplot(fig)



def ftu_tests(col, df, cols_lst):
    from scipy.stats import zscore, levene, ttest_ind, mannwhitneyu
    if col == 'Bankrupt?':
        return

        # Фильтрация данных
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
    f_stat, p_value = levene(bankrupt_group, non_bankrupt_group)
    if p_value > 0.05:
        t_stat, p_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=True)
        if p_value < 0.05:
            cols_lst.append(col)
        else:
            t_stat, p_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
            if p_value < 0.05:
                cols_lst.append(col)
            else:
                u_stat, p_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
                if p_value < 0.05:
                    cols_lst.append(col)
    else:
        t_stat, p_value = ttest_ind(bankrupt_group, non_bankrupt_group, equal_var=False)
        if p_value < 0.05:
            cols_lst.append(col)
        else:
            u_stat, p_value = mannwhitneyu(bankrupt_group, non_bankrupt_group, alternative='two-sided')
            if p_value < 0.05:
                cols_lst.append(col)





def filter_by_category(choice):
    if choice == 'Both':
        filtered_df = ftu_df
    elif choice == 'Bankrupt':
        filtered_df = ftu_df[ftu_df['Bankrupt?'] == 1]
    else:  # Non-Bankrupt
        filtered_df = ftu_df[ftu_df['Bankrupt?'] == 0]

    return filtered_df


# if bankrupt_button == 'Both':
#     wrapper_name = "Columns of statistical significant differences for bankrupts and non-bankrupts"
#
#     if not sorted_cols_lst:
#         st.error("No significant columns found.")
#     else:
#         # Инициализация списков для данных
#         features, categories, subcategories = [], [], []
#         values = []
#         details_category = []
#
#         for col in sorted_cols_lst:
#             corr_text = corr_column(col)  # Корреляционный текст
#             ftu_result = ftu_tests(col)  # Результаты теста
#
#             # Данные для верхнего уровня (Feature)
#             features.extend([wrapper_name] * 3)  # Привязка к категории
#             categories.extend([col] * 3)  # Колонка как категория
#             subcategories.extend([
#                 corr_text,
#                 bankrupt_disc_column(col),
#                 non_bankrupt_disc_column(col)
#             ])
#             values.extend([0.5, 0.25, 0.25])  # Пропорции: 50% категория, 25% подкатегории
#
#             # Добавляем детализированные данные для каждого уровня
#             details_category.extend([f"{col}<br>{ftu_result}"] * 3)
#
#         # Создаем DataFrame с деталями
#         tree_data = pd.DataFrame({
#             "Feature": features,
#             "Category": categories,
#             "Subcategory": subcategories,
#             "Value": values,
#             "Column": details_category
#         })
#
#         # Построение Treemap
#         fig = px.treemap(
#             tree_data,
#             path=["Feature", "Category", "Subcategory"],  # Иерархия уровней
#             values="Value",  # Размер узлов
#             custom_data=["Column"]  # Доп. данные для hover
#         )
#
#         # Настройка hovertemplate для разных уровней
#         fig.update_traces(
#             root_color="lightblue",  # Цвет корневого узла
#             marker=dict(line=dict(width=2, color="DarkSlateGray")),  # Стиль линий
#             hovertemplate="<b>%{label}</b><br>Column: %{customdata[0]}"  # Формат подсказок
#         )
#
#         # Отображение графика
#         st.plotly_chart(fig, use_container_width=True)

