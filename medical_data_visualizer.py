import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2: Add an overweight column to the data.
# BMI = weight (kg) / height (m)^2
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

# 3: Normalize data by making 0 always good and 1 always bad.
# If cholesterol or gluc > 1, set it to 1; otherwise, set to 0.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4: Draw the Categorical Plot in the draw_cat_plot function.
def draw_cat_plot():
    # 5: Create a DataFrame for the categorical plot using pd.melt.
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6: Group and reformat the data to split it by cardio.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')

    # 7: Convert the data into long format and create the catplot.
    fig = sns.catplot(data=df_cat, 
                      x='variable', 
                      y='total', 
                      hue='value', 
                      col='cardio', 
                      kind='bar').fig

    # 8: Save the plot to a file.
    fig.savefig('catplot.png')
    return fig

# 10: Draw the Heat Map in the draw_heat_map function.
def draw_heat_map():
    # 11: Clean the data in the df_heat variable by filtering out invalid data.
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Calculate the correlation matrix.
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15: Draw the heatmap.
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                square=True, 
                cbar_kws={'shrink': 0.5}, 
                center=0)

    # 16: Save the plot to a file.
    fig.savefig('heatmap.png')
    return fig
