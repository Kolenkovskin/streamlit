# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from numpy import random
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

exploring_header = "Exploring data"

df = pd.read_csv('./data.csv')

df

print('Common discription of data frame')
print()
print('Columns')
print()
print(df.columns)
print()
print('Common info')
print()
print(df.info())
print()
print('Common statistical view')
print()
print(df.describe())

df.isna().any()

df.columns = df.columns.str.strip()

# Проверяем уникальные значения в столбце 'Bankrupt?'
df['Bankrupt?'].unique()


df['Net Income Flag'].unique()

df['Liability-Assets Flag'].unique()

fixing_head = "Rationalizing and cleaning data"
print(fixing_head)

# Removing columns
df = df.drop(['Net Income Flag', 'Liability-Assets Flag'], axis=1)

# +
# Adding exclusions
exclude_columns = ['Bankrupt?']


# Decreasing data bit debth
df_converted = df.loc[:, ~df.columns.isin(exclude_columns)].astype('float32')

# Uniting back with columns excluded
df = pd.concat([df[exclude_columns], df_converted], axis=1)
# -

#df['Liability-Assets Flag'] = df['Liability-Assets Flag'].astype('int8')
df['Bankrupt?'] = df['Bankrupt?'].astype('int8')
#df['Net Income Flag'] = df['Net Income Flag'].astype('int8')

df.info()

df

intro = 'I am excited to present to you an interactive analysis of financial data using the Streamlit platform. This project showcases my skills in data analysis, visualization, and the ability to extract meaningful insights from complex datasets. The analysis is built around a dataset that evaluates financial indicators and identifies key differences between bankrupt and non-bankrupt companies. The analysis is deployed as an interactive Streamlit app. Users can dynamically explore the data, visualize trends, and better understand the critical financial features driving the results. This presentation demonstrates not only technical skills in Python, pandas, and visualization libraries like Matplotlib and Seaborn, but also the ability to communicate insights effectively. This project represents a practical application of my analytical and programming skills, providing a compelling demonstration of my capabilities in data analysis. I would be delighted to receive your feedback or discuss further how these insights can be applied to real-world scenarios. Thank you for your time and consideration.'

hist_box_head = "Visualizing distributions of financial indicators for bankrupt and non-bankrupt companies."
print(hist_box_head)

# +
import warnings
warnings.filterwarnings("ignore")

# List of columns representing financial indicators for analysis
financial_columns = [
    'ROA(A) before interest and % after tax',  # Sample column for analyzing asset profitability
    'Realized Sales Gross Margin',             # Sample column for analyzing gross margin
    'Net Value Per Share (B)'                  # Sample column for analyzing stock value
]

# Split data into bankrupt and non-bankrupt company groups
bankrupt_df = df[df['Bankrupt?'] == 1]
non_bankrupt_df = df[df['Bankrupt?'] == 0]

# Plot histograms and boxplots for each financial indicator
for column in financial_columns:
    plt.figure(figsize=(14, 10))

    # Histogram
    plt.subplot(2, 2, 1)
    sns.histplot(bankrupt_df[column], color='red', label='Bankrupt', kde=True)
    plt.subplot(2, 2, 2)
    sns.histplot(non_bankrupt_df[column], color='green', label='Non-Bankrupt', kde=True)
    plt.title(f'Histogram of {column}')
    plt.legend()

    # Boxplot
    plt.subplot(2, 2, 3)
    sns.boxplot(x='Bankrupt?', y=column, data=df, palette=['green', 'red'])
    plt.title(f'Boxplot of {column}')

    plt.tight_layout()
    plt.show()

# -

hist_box_disc = "**In this analysis, we visualized the distributions of selected financial indicators for companies that** **declared bankruptcy and those that did not. We used histograms and boxplots for each financial indicator** **to understand the differences in their distribution between the two groups.**\n\n**Code Explanation**\n\n**Dividing Data:**\n\n**We split the data into two subsets based on bankruptcy status: bankrupt_df for companies that declared** **bankruptcy and non_bankrupt_df for those that did not.**\n\n**Histograms:**\n\n**For each selected financial indicator, we plotted histograms of the indicator’s distribution, comparing** **the bankrupt and non-bankrupt groups.**\n\n**The histograms are colored red for bankrupt companies and green for non-bankrupt companies, allowing us to** **visually assess differences in the distribution of each metric.**\n\n**Boxplots:**\n\n**Additionally, we generated boxplots for each indicator, grouped by bankruptcy status. This helps us** **observe the central tendency and variability of the indicators within each group.****The boxplots show if there are significant outliers and allow a direct comparison of median values between** **the two groups.****Insights from the Results****ROA (Return on Assets): We observe a notable difference in the distribution of ROA between bankrupt and** **non-bankrupt companies. Non-bankrupt companies generally exhibit higher ROA values, indicating more** **efficient asset utilization in generating profit.**\n\n**Realized Sales Gross Margin: The distribution is similar between the two groups, but there is slightly** **more variability in non-bankrupt companies. This indicator does not show a clear distinction between** **bankrupt and non-bankrupt companies.**\n\n**Net Value Per Share: The distribution of this metric appears quite similar across both groups, with minor** **differences in variability. It suggests that net value per share may not be a strong** **differentiator between bankrupt and non-bankrupt companies.**\n\n**These visualizations help us quickly identify which financial indicators may be more indicative of bankruptcy risk and guide further statistical analysis. The boxplots specifically highlight if there are substantial differences in median values between the groups, aiding in understanding the central tendencies of these indicators across bankrupt and non-bankrupt companies.**"
hist_box_disc = hist_box_disc.replace('**', '')
print(hist_box_disc)

heatmap_head = "A correlation heatmap of relationships between various financial indicators"
print(heatmap_head)

# +
# Select columns representing financial indicators for analysis
# For example, metrics of profitability, margin, and other key indicators
financial_columns = [
    'ROA(A) before interest and % after tax',  # Example column for asset profitability analysis
    'Realized Sales Gross Margin',            # Example column for sales gross margin analysis
    'Net Value Per Share (B)',                 # Example column for stock price analysis
    'Current Ratio',                           # Current liquidity ratio
    'Quick Ratio',                             # Quick liquidity ratio
    'Equity to Liability'                      # Equity-to-liability ratio
]

# Create a new DataFrame with only the selected columns
correlation_df = df[financial_columns]

# Calculate the correlation matrix
correlation_matrix_compact = correlation_df.corr()
mask_cmc = np.triu(correlation_matrix_compact)

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_compact, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask_cmc)
plt.title('Correlation Heatmap of Financial Indicators')
plt.show()

# -

heatmap_disc = "The heatmap visually displays the strength of the correlations between these financial indicators: Quick Ratio and Current Ratio have a very high positive correlation (0.92), indicating that these two metrics are closely related and move in tandem, which is expected as both measure liquidity. Equity to Liability also has moderate correlations with both the Quick Ratio (0.76) and Current Ratio (0.71), suggesting that liquidity measures have a relationship with capital structure. Other indicators show weaker correlations, such as Realized Sales Gross  Margin and Net Value Per Share (B) having a moderate correlation of 0.53, suggesting a possible link between sales efficiency and stock value. This heatmap provides us with insights into which indicators are closely linked, guiding further analysis, such as evaluating how these relationships might differ between bankrupt and non-bankrupt companies."
heatmap_disc = heatmap_disc.replace('. ', '\n\n')
print(heatmap_disc)

lin_pair_head = "Pairs of features that have a strong linear relationship"

# Executing all numerical definitions excluding categorical definitions
numeric_df = df.drop(['Bankrupt?'], axis=1)

# Creating correlation matrix
correlation_matrix_total = numeric_df.corr()
correlation_matrix_total

# +
# Setting correlation thresholds
lower_threshold = 0.7
upper_threshold = 1.0

# Obtaining the correlation matrix and excluding the main diagonal (where correlation is 1)
correlation_matrix_no_diag = correlation_matrix_total.where(~np.eye(correlation_matrix_total.shape[0], dtype=bool))

# Filtering correlations within the range of 0.7 to 1
high_correlation = correlation_matrix_no_diag[(correlation_matrix_no_diag >= lower_threshold) & (correlation_matrix_no_diag < upper_threshold)]

# Formatting the output with feature names for rows and columns with high correlations
high_correlation_pairs = high_correlation.stack().reset_index()
high_correlation_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

# Displaying the result
high_correlation_pairs
# -

lin_pair_disc = 'Results Interpretation:****The resulting DataFrame, high_correlation_pairs, lists pairs of features with correlation values close to** **1, indicating strong linear relationships.****For example, "ROA(C) before interest and depreciation before interest" has a high correlation with both** **"ROA(A) before interest and % after tax" (0.940095) and "ROA(B) before interest and depreciation after** **tax" (0.986837), suggesting that these variations of Return on Assets (ROA) are strongly interrelated.****Other pairs, such as "Liability to Equity" and "Current Liabilities/Equity," also show high correlation** **(0.963919), which could imply that these indicators tend to move together due to their related financial** **meanings.****This analysis helps us identify which financial indicators are closely linked, which can be useful for simplifying the model or selecting features that convey unique information. High correlations may also indicate redundancy among features, allowing us to consider dropping some features or combining them in further analyses.**'
lin_pair_disc = lin_pair_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n')
print(lin_pair_disc)

corr_diff_head = 'A correlation differences between bankrupt and non-bankrupt companies'
print(corr_diff_head)

# Excluding categorical variables unused
#bunkrupticy_df = df.drop(['Net Income Flag','Liability-Assets Flag'], axis=1)

# +
# Create subsets for bankrupt and non-bankrupt companies
bankrupt_df = df[df['Bankrupt?'] == 1]
non_bankrupt_df = df[df['Bankrupt?'] == 0]

# Calculate the correlation matrix for both subsets
corr_matrix_bankrupt = bankrupt_df.corr()
corr_matrix_non_bankrupt = non_bankrupt_df.corr()

# Determine the difference in correlation between the two groups
correlation_diff = corr_matrix_bankrupt - corr_matrix_non_bankrupt

# Extract pairs with the most significant correlation differences
significant_correlation_diff = correlation_diff[(correlation_diff.abs() > 0.3)].stack().reset_index()
significant_correlation_diff.columns = ['Feature 1', 'Feature 2', 'Correlation Difference']

# Print significant differences
print("Significant correlation differences between groups:")
significant_correlation_diff
# -

corr_diff_disc = 'The table displays pairs of financial indicators with significant differences in correlation** **between bankrupt and non-bankrupt companies.****Each row represents a pair of features where the correlation difference meets the defined** **significance threshold (e.g., 0.3 in absolute value). Here’s a breakdown of some key insights:****High Positive Correlation Differences:****"ROA(C) before interest and depreciation before interest" and "Total Asset Return Growth Rate** **Ratio" show a correlation difference of 0.689, suggesting that in one group (likely** **non-bankrupt companies), these indicators are more closely related than in the other.****"Equity to Liability" and "Current Ratio" have a correlation difference of 0.866, which may** **imply that for non-bankrupt companies, the balance between equity and liability is more** **strongly associated with liquidity.****High Negative Correlation Differences:****"ROA(C) before interest and depreciation before interest" and "Total Asset Turnover" have a** **correlation difference of -0.418, indicating that this relationship is more pronounced in one** **group but negatively correlated in the other.****"Equity to Liability" and "Cash Flow to Liability" with a difference of -0.874 show a** **significant divergence, implying that these indicators behave differently in assessing** **financial stability across bankrupt and non-bankrupt groups.****These differences suggest that certain financial indicators and their relationships are differently influenced by the financial health of companies. Such insights could be valuable in distinguishing between stable and at-risk companies, as well as in understanding specific factors contributing to financial risk.**'
corr_diff_disc = corr_diff_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n')
print(corr_diff_disc)

# +
# Set the threshold for correlation significance
threshold = 0.5

# Filter rows where the correlation difference exceeds the threshold in absolute value
high_diff_pairs = significant_correlation_diff[significant_correlation_diff['Correlation Difference'].abs() > threshold]

# Display the result
print("Pairs of features with correlation differences above the specified threshold:")
high_diff_pairs

# -

corr_diff_thres_disc = '**Here are pairs of financial indicators where the correlation difference between bankrupt and****non-bankrupt companies is above the threshold of 0.5. This highlights relationships that behave very** **differently between the two groups, which could be critical indicators of financial distress.****For example:****The pair "ROA(C) before interest and depreciation before interest" and "Total Asset Return Growth Rate** **Ratio" has a high correlation difference of approximately 0.692, indicating a strong disparity in how** **these metrics relate across bankrupt and non-bankrupt companies.****Another pair, "ROA(A) before interest and % after tax" and "Total Asset Return Growth Rate Ratio," also shows a high correlation difference of around 0.685, further emphasizing the variance in how these metrics interact in different financial stability contexts.****These significant differences could serve as indicators for assessing financial health and predicting bankruptcy risk.**'
corr_diff_thres_disc = corr_diff_thres_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').strip()
print(corr_diff_thres_disc)

bank_dist_head = 'The distribution of some indicators (features) for bankrupt and non-bankrupt companies with significant correlation differences between the two groups.'
print(bank_dist_head)

# +
# Selecting significant features from previous calculations, for example, filtered by correlation or significance
features_to_plot = high_diff_pairs['Feature 1'].unique()[:10]  # Limit to 10 features for clarity

for feature in features_to_plot:
    # Создаем фигуру с двумя подграфиками
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    
    # Определяем размер выборки для банкротов
    bankrupt_sample_size = min(1000, len(df[df['Bankrupt?'] == 1]))
    sns.histplot(data=df[df['Bankrupt?'] == 1].sample(n=bankrupt_sample_size, replace=False), x=feature, kde=True, element='step', color='red', ax=axes[0])
    axes[0].set_title(f"Distribution of {feature} for Bankrupt Companies")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")

    # Определяем размер выборки для не банкротов
    non_bankrupt_sample_size = min(1000, len(df[df['Bankrupt?'] == 0]))
    sns.histplot(data=df[df['Bankrupt?'] == 0].sample(n=non_bankrupt_sample_size, replace=False), x=feature, kde=True, element='step', color='green', ax=axes[1])
    axes[1].set_title(f"Distribution of {feature} for Non-Bankrupt Companies")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# -

bank_dist_disc = '**The histograms show the distribution of values for both bankrupt and non-bankrupt companies, with a density curve (kde=True) overlaid for better visualization.This allows us to observe how certain indicators differ in their distribution between companies that went bankrupt and those that did not.****For example:****We can see differences in the shape and central tendency of distributions for bankrupt and non-bankrupt companies for metrics such as "ROA(C) before interest and depreciation before interest." Significant discrepancies in these distributions may indicate financial characteristics or thresholds that correlate strongly with bankruptcy risk.**'
bank_dist_disc = bank_dist_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').strip()
print(bank_dist_disc)

stat_diff_head = 'Insights into the statistical differences between bankrupt and non-bankrupt companies'
print(stat_diff_head)

# +
from scipy.stats import f_oneway, ttest_ind

# Splitting data into groups of bankrupt and non-bankrupt
group_bankrupt = df[df['Bankrupt?'] == 1]
group_non_bankrupt = df[df['Bankrupt?'] == 0]

# Creating lists for significant and non-significant features based on the t-test
significant_features = []
non_significant_features = []

# Creating lists to store F-test results (for checking equal variance)
equal_variance_features = []
unequal_variance_features = []

# Iterating over the features
for feature in features_to_plot:
    # F-test to check equal variance
    stat, p_value_f = f_oneway(group_bankrupt[feature].dropna(), group_non_bankrupt[feature].dropna())
    
    # Determining equal variance
    if p_value_f > 0.05:  # If p-value of F-test is greater than 0.05, variances are considered equal
        equal_variance_features.append(feature)
        equal_var = True
    else:  # If less than 0.05, variances are considered unequal
        unequal_variance_features.append(feature)
        equal_var = False

    # t-test considering equal or unequal variance
    stat, p_value_t = ttest_ind(group_bankrupt[feature].dropna(), group_non_bankrupt[feature].dropna(), equal_var=equal_var, nan_policy='omit')
    
    # Determining feature significance
    if p_value_t <= 0.05:
        significant_features.append(feature)
    else:
        non_significant_features.append(feature)

# Creating DataFrame with the results
df_results = pd.DataFrame({
    'Significant Features (p <= 0.05)': pd.Series(significant_features),
    'Non-Significant Features (p > 0.05)': pd.Series(non_significant_features),
    'Equal Variance Features': pd.Series(equal_variance_features),
    'Unequal Variance Features': pd.Series(unequal_variance_features)
})

# Displaying DataFrame
df_results

# -

stat_diff_disc = '**Significant Features (p <= 0.05)****These features have a p-value from the t-test less than or equal to 0.0001, indicating that the differences in the means between bankrupt and non-bankrupt companies are statistically significant. Interpreting these features: These metrics show significant variation between the two groups, suggesting that they could be useful indicators in distinguishing bankrupt from non-bankrupt companies.****Significant Features in this analysis:****ROA(C) before interest and depreciation before interestROA(A) before interest and % after taxROA(B) before interest and depreciation after taxCash flow rateNet Value Per Share (B)****2. Non-Significant Features (p > 0,05)****These features have a p-value greater than 0.0001, suggesting no statistically significant difference in the means between the two groups. Interpreting these features: These metrics do not show substantial differences between bankrupt and non-bankrupt companies. They might be less informative for predicting or differentiating bankruptcy risk.****Non-Significant Features in this analysis:****Operating Profit RatePre-tax net Interest Rate After-tax net Interest Rate Non-industry income and expenditure/revenue Continuous interest rate (after tax)****3. Equal Variance Features****Features in this column have passed the F-test with a p-value greater than 0.05, meaning that there is no evidence to reject the null hypothesis of equal variances between the two groups. Interpreting these features: These features can be analyzed using a t-test assuming equal variances, which is generally more robust.****Equal Variance Features in this analysis:****Operating Profit RatePre-tax net Interest RateAfter-tax net Interest RateNon-industry income and expenditure/revenueContinuous interest rate (after tax)****5. Unequal Variance Features****Features in this column did not pass the F-test (p-value <= 0.05), indicating that there is a significant difference in variances between the two groups. Interpreting these features: These features should be analyzed using a t-test that does not assume equal variances (Welch’s t-test), as the variance between groups differs.****Unequal Variance Features in this analysis:****ROA(C) before interest and depreciation before interes ROA(A) before interest and % after tax ROA(B) before interest and depreciation after tax Cash flow rate Net Value Per Share (B)****Summary Interpretation:****Significant Features indicate metrics that are markedly different between bankrupt and non-bankrupt companies, which could be valuable in developing predictive models or assessing bankruptcy risk. Equal and Unequal Variance Features guide us on which type of t-test is appropriate for each feature. The Non-Significant Features may not contribute as much to distinguishing between the groups and could be less relevant in further analyses. This analysis helps in identifying key financial indicators that differentiate bankrupt from non-bankrupt companies, assisting in risk assessment and predictive modeling.**'
stat_diff_disc = stat_diff_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').replace('0.0001', '0.05').replace('0,05', '0.05').strip()
print(stat_diff_disc)

stat_feat_head = 'Visualization of distribution of each significant feature separately for bankrupt and non-bankrupt companies.'
print(stat_feat_head)

# +
# Plot significant features
for feature in significant_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Subplot for bankrupt companies
    sns.histplot(data=df[df["Bankrupt?"] == 1], x=feature, kde=True, element="step", color="red", ax=axes[0])
    axes[0].set_title(f"Distribution of Significant Feature: {feature} (Bankrupt)")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")
    
    # Subplot for non-bankrupt companies
    sns.histplot(data=df[df["Bankrupt?"] == 0], x=feature, kde=True, element="step", color="blue", ax=axes[1])
    axes[1].set_title(f"Distribution of Significant Feature: {feature} (Non-Bankrupt)")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Plot non-significant features
for feature in non_significant_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    # Subplot for bankrupt companies
    sns.histplot(data=df[df["Bankrupt?"] == 1], x=feature, kde=True, element="step", color="red", ax=axes[0])
    axes[0].set_title(f"Distribution of Non-Significant Feature: {feature} (Bankrupt)")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Frequency")
    
    # Subplot for non-bankrupt companies
    sns.histplot(data=df[df["Bankrupt?"] == 0], x=feature, kde=True, element="step", color="blue", ax=axes[1])
    axes[1].set_title(f"Distribution of Non-Significant Feature: {feature} (Non-Bankrupt)")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()


# -

stat_feat_head = '**These visualizations illustrate that bankrupt companies generally have lower values in key financial indicators, while non-bankrupt companies have higher, more concentrated distributions. These patterns suggest that these financial metrics are potential indicators of financial stability, with higher values possibly indicating a lower likelihood of bankruptcy. Each feature`s distinct distribution patterns between the two groups reinforce their significance as distinguishing features.**'
stat_feat_head = stat_feat_head.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').strip()
print(stat_feat_head)

ftest_boxs_head = 'Distributions of financial features between bankrupt and non-bankrupt companies based on the F-test results.'
print(ftest_boxs_head)

# +
# Plot equal variance features
for feature in equal_variance_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Bankrupt?", y=feature, palette="coolwarm")
    plt.title(f"Boxplot of Equal Variance Feature: {feature}")
    plt.xlabel("Bankrupt?")
    plt.ylabel(feature)
    plt.xticks([0, 1], ["Non-Bankrupt", "Bankrupt"])
    plt.show()

# Plot unequal variance features
for feature in unequal_variance_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Bankrupt?", y=feature, palette="coolwarm")
    plt.title(f"Boxplot of Unequal Variance Feature: {feature}")
    plt.xlabel("Bankrupt?")
    plt.ylabel(feature)
    plt.xticks([0, 1], ["Non-Bankrupt", "Bankrupt"])
    plt.show()

# -

ftest_boxs_disc = '**Equal Variance Features: These features have similar distributions for both bankrupt and non-bankrupt companies, suggesting that they might not be as useful in distinguishing between the two groups.****Outliers: Both groups contain outliers in these features, indicating that a subset of companies has atypical values, regardless of their bankruptcy status.****These visualizations help to confirm that equal variance features likely contribute less to predicting bankruptcy, while unequal variance features might show more pronounced differences.**'
ftest_boxs_disc = ftest_boxs_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').strip()
print(ftest_boxs_disc)

stat_sum_head = 'Summary of statistical significance and variance equality for various features when comparing bankrupt and non-bankrupt companies.'
print(stat_sum_head)

# +
summary_data = {
    "Significant": [1 if feature in significant_features else 0 for feature in df_results["Significant Features (p <= 0.05)"].dropna().tolist() + df_results["Non-Significant Features (p > 0.05)"].dropna().tolist()],
    "Equal Variance": [1 if feature in equal_variance_features else 0 for feature in df_results["Equal Variance Features"].dropna().tolist() + df_results["Unequal Variance Features"].dropna().tolist()]
}
summary_df = pd.DataFrame(summary_data, index=df_results["Significant Features (p <= 0.05)"].dropna().tolist() + df_results["Non-Significant Features (p > 0.05)"].dropna().tolist())

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(summary_df, annot=True, cmap="YlGnBu", cbar=True, xticklabels=["Significant (p <= 0.05)", "Equal Variance"], yticklabels=summary_df.index)
plt.title("Summary of Statistical Significance and Variance Equality for Features")
plt.show()

# -

stat_sum_disc = '**Important Features: Features that are statistically significant and have unequal variance may serve as strong indicators for differentiating bankrupt from non-bankrupt companies. These features vary not only in mean but also in spread, indicating that they capture more nuanced differences in financial health.****Less Informative Features: Non-significant features are less relevant for predicting bankruptcy risk, as they do not show meaningful differences between the groups.****This heatmap effectively provides a quick overview of which features are most informative and should be prioritized in any predictive model for bankruptcy analysis.**'
stat_sum_disc = stat_sum_disc.replace('**', ' ').replace('  ', ' ').replace('. ', '.\n\n').strip()
print(stat_sum_disc)

fin_conc = 'Final Conclusions'
print(fin_conc)

# +
# Key findings based on correlation analysis
print("1. Correlation analysis showed that the following financial indicators have a strong correlation with each other and may be indicators of the financial stability of the company:\n\nBelow is a list of the strongest correlation pairs:")

for index, row in high_correlation_pairs.iterrows():
    print(f"- {row['Feature 1']} with {row['Feature 2']} (Correlation: {row['Correlation']:.2f})")
# -

# Findings based on the differences between bankrupt and non-bankrupt companies
print("2. Analysis of differences between bankrupt and non-bankrupt companies:\n\nThe most significant differences are observed in the following indicators:")
for feature in significant_features:
    print(f"- {feature}")

fin_conc_disc = 'These differences can be used to assess the likelihood of bankruptcy and financial stability.'
print(fin_conc_disc)

rec_head = 'Recommendations for analysis improvement'

rec_disc = '1. Adding time series data: If time series data were available, it would be possible to study the dynamics of key indicators over time and identify trends preceding bankruptcy. \n2. Additional financial metrics: Including additional financial metrics, such as cash flow, debt data, and capital expenditures, could deepen the analysis. \n3. External economic data: Adding data on market conditions, such as interest rates or inflation, could help explain the financial performance of companies.'
print(rec_disc)


