import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from scipy import stats
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.data import load_clean_data

pink, mauve, plum, taupe, cream, ink = "#D98AA0", "#C98BAE", "#9B6A8F", "#CBB293", "#FBF3EC", "#6E4B5E"
palette = ["#E7A6B0", "#D98AA0", "#C98BAE", "#E4D2B8", "#CBB293", "#9B6A8F"]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = cycler(color=palette)
plt.rcParams["figure.facecolor"] = cream
plt.rcParams["axes.facecolor"] = cream
plt.rcParams["savefig.facecolor"] = cream

st.set_page_config(page_title="Social media addiction analysis", page_icon="🌸", layout="centered")
st.markdown(f"<style>.stApp{{background-color:{cream};}}h1,h2,h3{{color:{ink};}}</style>", unsafe_allow_html=True)

data = st.cache_data(load_clean_data)()

def show(fig):
    st.pyplot(fig)
    plt.close(fig)

st.title("Project: TikTok and Instagram addiction analysis")
st.write("Course: Python for Data Science, DSBA, 2025/2026.")
st.write("Authors: Mitricheva Anna and Romanovskaia Eva, group 251.")
st.write("We looked at 10000 social media users and found that the amount of time spent on social media is the key factor that affects the addiction score.")
st.write("Shortly, TikTok is slightly more connected to addiction than Instagram. Age and sleep, surprisingly, do not seem to matter at all.")

st.header("Setup")

st.header("Data")
st.write("Dataset is from https://www.kaggle.com/datasets/abdulmaliklodhra/tiktok-and-instagram-addiction-dataset-20152060")
st.dataframe(data.head(20))
c1, c2 = st.columns(2)
c1.metric("Rows", len(data))
c2.metric("Columns", data.shape[1])
st.write("The main fields we use:")
st.write("1. tiktok_minutes_daily and instagram_minutes_daily — how many minutes per day a person spends on each app")
st.write("2. night_usage_ratio — what share of their usage happens at night, from 0 to 1")
st.write("3. attention_span_score — a measure of attention span")
st.write("4. dopamine_dependency_score — how dependent the user is on dopamine hits")
st.write("5. sleep_hours — how many hours they sleep")
st.write("6. age, country, year — basic information about who the user is")
st.write("7. addiction_score — the numeric addiction score")
st.write("8. addiction_level — the addiction level: Low, Medium, High or Severe")

st.header("Description")
st.write("Looking at the details:")
fields = ["tiktok_minutes_daily", "instagram_minutes_daily", "sleep_hours", "addiction_score", "night_usage_ratio", "attention_span_score"]
desc = data[fields].describe().T
desc["median"] = data[fields].median()
st.dataframe(desc[["count", "mean", "median", "std", "min", "25%", "75%", "max"]].round(2))
st.write("On average users spend about 121 minutes a day on TikTok and about 100 on Instagram. Sleep is around 7 hours and the addiction score is around 58. Night usage is spread between 0 and 1.")

st.header("Data cleanup")
st.write("Our data is already clean, so we just showed it.")
c1, c2, c3 = st.columns(3)
c1.metric("Missing values", int(data.isna().sum().sum()))
c2.metric("Duplicates", int(data.duplicated().sum()))
c3.metric("Shape", f"{data.shape[0]} x {data.shape[1]}")

st.header("Data transformation")
st.write("New columns:")
st.write("1. total_minutes — tiktok_minutes_daily plus instagram_minutes_daily")
st.write("2. daily_hours — total_minutes divided by 60")
st.write("3. tiktok_share — tiktok_minutes_daily divided by total_minutes")
st.write("4. night_minutes — total_minutes multiplied by night_usage_ratio")
st.write("5. age_group — age split into three groups: under 25, 25 to 35, and over 35")
st.write("6. heavy_user — equals 1 if total time is above the median, otherwise 0")
st.write("7. level_num — addiction level converted to a number from 1 (Low) to 4 (Severe)")
st.dataframe(data[["total_minutes", "daily_hours", "tiktok_share", "night_minutes", "age_group", "heavy_user", "level_num"]].head(15))

st.header("Normalization")
st.write("The numeric fields are on very different scales (minutes go into the hundreds while ratios stay between 0 and 1). To make them comparable we standardize the columns to mean 0 and standard deviation 1 and keep the original columns for the readable plots.")
zcols = [c + "_z" for c in ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_minutes", "sleep_hours", "addiction_score"]]
st.dataframe(data[zcols].describe().round(2).T[["mean", "std", "min", "max"]])

st.header("Graphs")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
specs = [("tiktok_minutes_daily", "TikTok minutes per day", pink, "Minutes per day"), ("instagram_minutes_daily", "Instagram minutes per day", mauve, "Minutes per day"), ("addiction_score", "Addiction score", plum, "Score (0 to 100)"), ("sleep_hours", "Sleep hours", taupe, "Hours")]
for ax, (col, title, color, xlabel) in zip(axes.flat, specs):
    sns.histplot(data[col], bins=40, kde=True, color=color, ax=ax, edgecolor="white", linewidth=1.2)
    ax.axvline(data[col].mean(), color=ink, ls="--", lw=1.2, label=f"Mean: {data[col].mean():.1f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.legend()
show(fig)

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=data, x="addiction_level", y="total_minutes", order=["Low", "Medium", "High", "Severe"], hue="addiction_level", palette=palette[:4], legend=False, ax=ax)
ax.set_title("Total daily minutes by addiction level")
ax.set_xlabel("Addiction level")
ax.set_ylabel("Total minutes per day")
show(fig)

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.regplot(data=data.sample(3000, random_state=1), x="total_minutes", y="addiction_score", scatter_kws={"alpha": 0.35, "s": 16, "color": pink}, line_kws={"color": plum, "lw": 2}, ax=ax)
ax.set_title("Total daily minutes vs addiction score")
ax.set_xlabel("Total minutes per day")
ax.set_ylabel("Addiction score")
show(fig)
st.write("The box plot shows total minutes rising sharply with the addiction level, and the scatter shows a clear upward trend between minutes and the score.")

corr_cols = ["tiktok_minutes_daily", "instagram_minutes_daily", "total_minutes", "night_usage_ratio", "night_minutes", "sleep_hours", "age", "addiction_score"]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(data[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdPu", vmin=-1, vmax=1, linewidths=0.5, linecolor="white", ax=ax)
ax.set_title("Correlation matrix")
show(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, col, title, color in zip(axes, ["tiktok_minutes_daily", "instagram_minutes_daily"], ["TikTok", "Instagram"], [pink, mauve]):
    sns.regplot(data=data.sample(3000, random_state=2), x=col, y="addiction_score", ax=ax, scatter_kws={"alpha": 0.3, "s": 14, "color": color}, line_kws={"color": plum, "lw": 2})
    r = stats.pearsonr(data[col], data["addiction_score"])[0]
    ax.set_title(f"{title}: r = {round(r, 3)}")
    ax.set_xlabel(f"{title} minutes per day")
    ax.set_ylabel("Addiction score")
show(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=data, x="age_group", y="addiction_score", order=["<25", "25-35", "35+"], hue="age_group", palette=palette[:3], legend=False, ax=ax)
ax.set_title("Addiction score by age group")
ax.set_xlabel("Age group")
ax.set_ylabel("Addiction score")
show(fig)
st.dataframe(data.groupby("age_group", observed=True)["addiction_score"].mean().round(2))

profile = data.groupby("addiction_level", observed=True).agg(n=("addiction_level", "size"), total_minutes=("total_minutes", "mean"), night_minutes=("night_minutes", "mean"), sleep_hours=("sleep_hours", "mean"), age=("age", "mean")).round(1)
st.dataframe(profile)
st.write("The correlation matrix makes the picture clear: the addiction score is strongly linked to total minutes and to the TikTok and Instagram minutes, while sleep and age do not. TikTok has a higher correlation than Instagram. The age groups have almost the same average score, and the profile table shows that higher addiction levels mostly mean a lot more minutes spent.")

st.header("Hypotheses")

st.subheader("1. Time drives addiction, and TikTok matters more than Instagram")
r_tt, p_tt = stats.pearsonr(data["tiktok_minutes_daily"], data["addiction_score"])
r_ig, p_ig = stats.pearsonr(data["instagram_minutes_daily"], data["addiction_score"])
r_tot, p_tot = stats.pearsonr(data["total_minutes"], data["addiction_score"])
X = np.column_stack([np.ones(len(data)), data["tiktok_minutes_daily_z"], data["instagram_minutes_daily_z"]])
beta = np.linalg.lstsq(X, data["addiction_score_z"].values, rcond=None)[0]
c1, c2, c3 = st.columns(3)
c1.metric("r TikTok", round(r_tt, 3))
c2.metric("r Instagram", round(r_ig, 3))
c3.metric("r Total", round(r_tot, 3))
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(x=["TikTok", "Instagram", "Total"], y=[r_tt, r_ig, r_tot], hue=["TikTok", "Instagram", "Total"], palette=palette[:3], legend=False, ax=ax)
ax.set_title("Correlation with the addiction score")
ax.set_ylabel("Pearson r")
ax.set_ylim(0, 1)
for i, v in enumerate([r_tt, r_ig, r_tot]):
    ax.text(i, v + 0.02, round(v, 2), ha="center", color=ink)
show(fig)
st.write(f"The hypothesis holds. Both apps have a strong and significant correlation with the addiction score (p well below 0.05), the combined time is even more - with r around 0.85, and the regression gives TikTok a larger coefficient ({round(beta[1], 3)}) than Instagram ({round(beta[2], 3)}). So time clearly drives the score and TikTok matters a bit more, which we can also see in the two scatter plots above.")

st.subheader("2. Age and sleep are not related to the addiction score")
st.write("A common idea is that young people or people who sleep less are more addicted. We did not expect the opposite to show in this data, that age and sleep have no real link to the score. We test both correlations and compare the youngest users (under 25) with the oldest (over 35).")
r_age, p_age = stats.pearsonr(data["age"], data["addiction_score"])
r_sleep, p_sleep = stats.pearsonr(data["sleep_hours"], data["addiction_score"])
young = data[data["age"] < 25]["addiction_score"]
adult = data[data["age"] >= 35]["addiction_score"]
t, p_grp = stats.ttest_ind(young, adult, equal_var=False)
c1, c2 = st.columns(2)
c1.metric("r age", round(r_age, 3), f"p = {round(p_age, 3)}")
c2.metric("r sleep", round(r_sleep, 3), f"p = {round(p_sleep, 3)}")
st.write(f"Mean score, young: {round(young.mean(), 2)}, adults: {round(adult.mean(), 2)}. t-test young vs adults: t = {round(t, 2)}, p = {round(p_grp, 3)}.")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
s = data.sample(3000, random_state=4)
for ax, col, title in zip(axes, ["age", "sleep_hours"], ["Age", "Sleep hours"]):
    sns.regplot(data=s, x=col, y="addiction_score", ax=ax, scatter_kws={"alpha": 0.25, "s": 14, "color": taupe}, line_kws={"color": plum, "lw": 2})
    r = stats.pearsonr(data[col], data["addiction_score"])[0]
    ax.set_title(f"{title}: r = {round(r, 3)}")
    ax.set_xlabel(title)
    ax.set_ylabel("Addiction score")
show(fig)
st.write("The hypothesis holds. Both correlations are basically zero and not significant, and the regression lines are almost flat. So in this data age and sleep do not explain the addiction score.")

st.header("Conclusion")
st.write("1. The time spent on social media is the main thing linked to the addiction score, with the total daily minutes reaching a correlation of about 0.85.")
st.write("2. TikTok is connected to the score a little more than Instagram, both in the correlations and in the regression.")
st.write("3. Age and sleep are not related to the addiction score at all, which is a clear result against the usual assumption.")
st.write("4. Higher addiction levels mostly mean a lot more daily and night minutes, while sleep and age do not really matter here.")
st.write("What stands over the facts and graphs is that if we want to understand a person's addiction score in this data, it will be useful to know how much and how late they scroll, especially on TikTok, and not their age or how long they sleep. The difference between age groups is smaller than we initially assumed.")
st.write("")
st.write("P.S. One limitation of the dataset is that it contains self-reported behaviour, which may not always be fully accurate. The dataset also represents a specific sample of users, therefore the results should not be generalized to all social media users.")


st.header("Predict your addiction score")
st.write("Move the sliders and see the predicted addiction score based on our regression model.")

tt_min = st.slider("TikTok minutes per day", 0, int(data["tiktok_minutes_daily"].max()), 120)
insta_min = st.slider("Instagram minutes per day", 0, int(data["instagram_minutes_daily"].max()), 100)

tt_mean, tt_std = data["tiktok_minutes_daily"].mean(), data["tiktok_minutes_daily"].std()
insta_mean, insta_std = data["instagram_minutes_daily"].mean(), data["instagram_minutes_daily"].std()
score_mean, score_std = data["addiction_score"].mean(), data["addiction_score"].std()

tt_z = (tt_min - tt_mean) / tt_std
insta_z = (insta_min - insta_mean) / insta_std

X_pred = np.column_stack([np.ones(len(data)), data["tiktok_minutes_daily_z"], data["instagram_minutes_daily_z"]])
beta_pred = np.linalg.lstsq(X_pred, data["addiction_score_z"].values, rcond=None)[0]

pred_z = beta_pred[0] + beta_pred[1] * tt_z + beta_pred[2] * insta_z
pred_score = pred_z * score_std + score_mean
pred_score = float(np.clip(pred_score, 0, 100))

st.metric("Predicted addiction score", round(pred_score, 1))

if pred_score < 40:
    level = "Low"
elif pred_score < 60:
    level = "Medium"
elif pred_score < 80:
    level = "High"
else:
    level = "Severe"
st.write(f"Estimated addiction level: **{level}**")
st.progress(int(pred_score))


st.header("Explore by addiction level")
st.write("Pick an addiction level and see the average profile of these users.")

level_choice = st.radio("Choose addiction level:", ["Low", "Medium", "High", "Severe"], horizontal=True)

subset = data[data["addiction_level"] == level_choice]

c1, c2, c3 = st.columns(3)
c1.metric("Users", len(subset))
c2.metric("Avg total minutes", round(subset["total_minutes"].mean(), 1))
c3.metric("Avg sleep hours", round(subset["sleep_hours"].mean(), 1))

st.write(f"Average profile of **{level_choice}** users:")
st.dataframe(subset[["tiktok_minutes_daily", "instagram_minutes_daily", "night_minutes", "sleep_hours", "age"]].mean().round(1))