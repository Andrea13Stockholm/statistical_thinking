
##Ideal steps
##1. Design the experiment
##2. Data preparation
##3. Visualizingn the results
##4. HP testing & conclusions.

##In this example: 2 versions of webpage (old vs new). We want to increase conversion rate.


##Ho : p = p_0 versus H1 : p!= p_0.
## p_value < alpha, reject Ho. Alpha = 0.05. Confidence 1-alpha
## Power = 1-beta = probability of detecting a differential effects in the two groups when
## there is
## Effect size = the minimum differential effects that we do expect to see in the two groups.

##Control (C): we have shown them the old version; Treatment (T): we have shown them the new
##version. 0 if they did not convert; 1 if they converted.


import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# Link to Python documentation for statistical modelling.
# https://www.statsmodels.org/stable/user-guide.html#statistics-and-tools

# Power study
def size_of_each_group(p_ex_ante, p_ex_post, power, alpha, ratio_groups):

    # The effect size calculations depends on the target test, here I am comparing
    # proportions.

    effect_size = sms.proportion_effectsize(p_ex_ante, p_ex_post, method="normal")

    # The power calculations depends also on type of test
    # Z-test for 2 independent samples: NormalIndPower()

    required_n_raw = sms.NormalIndPower().solve_power(
        effect_size=effect_size, power=power, alpha=alpha, ratio=ratio_groups
    )

    required_n = ceil(required_n_raw)

    print("Needed observations (at least) for each group: " + str(required_n))
    return required_n, effect_size


# Interpreation of power: if there exist an actual difference in conversion rate,
# assuming the different is the one estimated, then we have power % chances to detect it.

required_n, effect_size = size_of_each_group(0.13, 0.15, 0.8, 0.05, 1)

##I used a prepared dataset from Kaggle

df = pd.read_csv("ab_data.csv")
df.head()

df.info()

df.groupby(["group"])["user_id"].count()

pd.crosstab(df["group"], df["landing_page"])

# Are there users appearing multiple times in the dataset?

session_counts = df["user_id"].value_counts(ascending=False)

print(
    "The number of users appearing multiple times in the dataset/number of show-ups is the following:"
)

print(session_counts[session_counts > 1].describe())

# Are these users showing up twice evenly across groups?
multiple_users = df[df["user_id"].isin(session_counts[session_counts > 1].index)].copy()
pd.pivot_table(
    multiple_users,
    values="timestamp",
    index="user_id",
    columns="group",
    aggfunc="count",
    margins=True,
).fillna(0)


# Why we need to delete duplicate users to interpret correctly the effects?

# There are users showing up twice in treatment:
display(df[df["user_id"] == 630052])

# There are users showing up twice in the control
display(df[df["user_id"] == 630137])


# If we assume that we want to randomize the session level, then users can come back and be
# randomly assigned to a landing_page.
# Assume that we do have two user types: Impulsive (I) & Non Impulsive (NI).
# Impulsive types will visit the page once and make a decision, so they won't come back.
# So far so good. But what about Non Impulsive onse?
# Suppose to consider a Non Impulsive user (NI) who visits 2 times the page before
# converting/making a decision. Since we randomize at session level, the same user can
# be exposed to different pages. Suppose this user in visit 1 get into the new page & she
# liked it a lot but the purchase is made when she visited the second time and the old
# page was shown. In this case, we have a misattribution of the positive effect.

# If we assume that sessions are independent (i.e. the experience of session n
# does not have an impact on session n+1), then we can keep double observations.

# But I do not believe that in the real world, sessions are independent: here again
# you need to assume what happens when NI types randomly assigned to new_page and then
# permanentely assign to that for every successive visit. How to fix it? Let's assume
# user level randomization (users are independent, meaning that one user's decision does not
# affect others' decisions to convert).

# Sessions not independent -> carry over effects -> randomize at user level;

# So, I assume that the randomization is at user level effect.


# Removing duplicate users
df = df[~df["user_id"].isin(multiple_users.user_id.to_list())]


# We need to randomly sample users now, stratifying at group level
control_sample = df[df["group"] == "control"].sample(n=required_n, random_state=22)
treatment_sample = df[df["group"] == "treatment"].sample(n=required_n, random_state=22)
ab_test_df = pd.concat([control_sample, treatment_sample], axis=0)
ab_test_df.reset_index(drop=True, inplace=True)

ab_test_df.info()

# Now we have a balanced and stratified sample, across groups, assuming that
# randomization is done at user level (sessions are not independent)
ab_test_df.groupby(["group", "landing_page"])["user_id"].count()


# How conversation rate looks like?

conversion_rates = ab_test_df.groupby(["group"])["converted"]

# Standard devitions and errors as in the population, since we have 4,720 observations
std_p = lambda x: np.std(x, ddof=0)
se_p = lambda x: np.std(x, ddof=0)

conversation_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversation_rates.columns = ["conversion_rate", "std_deviation", "std_error"]


# Let's plot our results:

g = sns.catplot(
    data=ab_test_df, x="group", y="converted", kind="bar", ci=95, aspect=0.95
)
plt.title("Conversation Rates (AB testing)")
plt.show()

# By just looking at the catplot, it does not seem that there is any difference. Let's try
# to do it from a statistical point of view.

from statsmodels.stats.proportion import proportions_ztest, proportion_confint


def result_subgroups(df, group, target):

    results = df[df["group"] == group][target]
    count = results.count()

    success = results.sum()

    return results, count, success

control_res, n_control, success_control = result_subgroups(
    ab_test_df, "control", "converted"
)
treat_res, n_treat, success_treat = result_subgroups(
    ab_test_df, "treatment", "converted"
)

# Let's stack results for using them in the Z testing now.
success = [success_control, success_treat]
nobs = [n_control, n_treat]

# I can derive z stat, p value and CI
z_stat, pval = proportions_ztest(success, nobs)

(lower_control, upper_control), (lower_treat, upper_treat) = proportion_confint(
    success, nobs, alpha=0.05
)

print(f"Z statistic value:  {z_stat:.5f}")
print(f"P_value:  {pval:.5f}")


if pval < 0.05:
    print("Reject the null")
else:
    print("We fail to reject the null")


print(f"95% CI for control:    {lower_control:.5f}, {upper_control:.5f} ")
print(f"95% CI for treatment:  {lower_treat:.5f}, {upper_treat:.5f}")

# Drawing conclusions
print("Our new design performes poorly")

# End-------------Analysis finished--------
