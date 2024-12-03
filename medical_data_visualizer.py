import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importing data and add overweight column
# BMI = weight in kg / height in m^2
df = pd.read_csv('medical_examination.csv')
df['overweight'] = ((df['weight'] / (df['height'] / 100) ** 2) > 25).astype(int)

# Normalising data by making 0 always good and 1 always bad
# If cholesterol or gluc > 1, set it to 1; otherwise, set to 0
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# Drawing cat plot
def draw_cat_plot():
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Grouping to start reformatting the data to split it by cardio
    # And finally converting the data into long format to create the catplot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().reset_index(name='total')
    fig = sns.catplot(data=df_cat, 
                      x='variable', 
                      y='total', 
                      hue='value', 
                      col='cardio', 
                      kind='bar').fig

    fig.savefig('catplot.png')
    return fig

# Drawing heat map
def draw_heat_map():
    # Cleaning the data and finding the correlation matrix
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    corr = df_heat.corr()
    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                square=True, 
                cbar_kws={'shrink': 0.5}, 
                center=0)

    fig.savefig('heatmap.png')
    return fig
