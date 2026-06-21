# Practical 4: Statistical Analysis using Python

import pandas as pd
import scipy.stats as stats

df = pd.read_csv("datasets/data.csv")

df["Calories"] = df["Calories"].fillna(df["Calories"].mean())


print("DESCRIPTIVE STATISTICS\n")

print("Mean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nMode:")
print(df.mode().iloc[0])

print("\nVariance:")
print(df.var(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

print("\nCORRELATION MATRIX\n")
print(df.corr(numeric_only=True))

group1 = df["Pulse"][:80]
group2 = df["Pulse"][80:]

t_stat, p_value = stats.ttest_ind(group1, group2)

print("\nT-TEST")
print("T-Statistic =", t_stat)
print("P-Value =", p_value)

if p_value < 0.05:
    print("Result: Significant Difference")
else:
    print("Result: No Significant Difference")


df["Pulse_Category"] = pd.cut(df["Pulse"],bins=3,labels=["Low", "Medium", "High"])

df["Duration_Category"] = pd.cut(df["Duration"],bins=3,labels=["Short", "Medium", "Long"])

contingency_table = pd.crosstab(df["Pulse_Category"],df["Duration_Category"])

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\nCHI-SQUARE TEST")
print("Chi-Square Statistic =", chi2)
print("P-Value =", p)

if p < 0.05:
    print("Result: Variables are Associated")
else:
    print("Result: Variables are Independent")

group_a = df[df["Duration"] <= 45]["Calories"]
group_b = df[(df["Duration"] > 45) & (df["Duration"] <= 60)]["Calories"]
group_c = df[df["Duration"] > 60]["Calories"]

f_stat, p_value = stats.f_oneway(group_a,group_b,group_c)

print("\nANOVA TEST")
print("F-Statistic =", f_stat)
print("P-Value =", p_value)

if p_value < 0.05:
    print("Result: Significant Difference Between Groups")
else:
    print("Result: No Significant Difference Between Groups")

print("\nStatistical Analysis Completed Successfully!")