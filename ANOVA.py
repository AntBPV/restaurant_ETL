import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Change this file name to the one you get from running ETL.py
# I renamed mine to what you see below.
df = pd.read_csv("restaurant_customer_satisfaction_clean.csv")

if 'AverageSpend' not in df.columns or 'PreferredCuisine' not in df.columns:
    raise ValueError("Columns not present on CSV file")

data = df[['AverageSpend', 'PreferredCuisine', 'VisitFrequency']].dropna()

model = ols('AverageSpend ~ C(PreferredCuisine) * C(VisitFrequency)', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("=== Resultados del test ANOVA ===")
print(anova_table)

p_value = anova_table["PR(>F)"].iloc[0]
if p_value < 0.05:
    print("\nResultado: Existe diferencia significativa entre los grupos (p < 0.05)")
else:
    print("\nResultado: No se encontraron diferencias significativas entre los grupos (p ≥ 0.05)")

if p_value < 0.05:
    print("\n=== Test Post-hoc: Tukey HSD ===")
    tukey = pairwise_tukeyhsd(endog=data['AverageSpend'],
                              groups=data['PreferredCuisine'],
                              alpha=0.05)
    print(tukey)