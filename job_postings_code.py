import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize, MWETokenizer
# nltk.download('punkt') # only need to run once

### Upload the file and pre-clean it:
jobs_all = pd.read_csv('gsearch_jobs.csv').replace("'", "", regex=True)
jobs_all.date_time = pd.to_datetime(jobs_all.date_time)
jobs_all = jobs_all.drop(labels=['Unnamed: 0', 'index'], axis=1, errors='ignore')
jobs_all.description_tokens = jobs_all.description_tokens.str.strip("[]").str.split(",")
jobs_all.location = jobs_all.location.str.strip(" ")

### Data Integrity check:
first_date = jobs_all.date_time.dt.date.min()
today_date = datetime.date.today()
date_count = pd.DataFrame(jobs_all.date_time.dt.date.value_counts())
missing_dates = list(pd.date_range(start=first_date, end=today_date).difference(date_count.index))
if len(missing_dates) > 0:
    print("Missing data for following dates:")
    for date in missing_dates:
        pass
        print(date)
else:
    pass
    print(f"No missing dates of data since inception of: {first_date}")

delta_days = (today_date - (first_date - datetime.timedelta(days=2))).days  # first day was actually day prior but UTC
jobs_day = round(len(jobs_all) / delta_days)

print(f"Average number of jobs per day: {jobs_day}")
print(f"Collecting data for {delta_days} days now...")

### Exploratory Data Analysis:


def eda_plot(column, topn=10):
    jobs_all[column].value_counts().nlargest(topn).plot(kind='bar')
    plt.title(f"'{column}' column value counts")
    plt.ylabel("Counts")
    plt.xticks(rotation=15, ha='right')
    plt.show()


columns = ['title', 'company_name', 'location', 'via', 'schedule_type', 'work_from_home']

for column in columns:
    eda_plot(column)

### Tokenizing keywords:
keywords_programming = [
    'sql', 'python', 'r', 'c', 'c#', 'javascript', 'js', 'java', 'scala', 'sas', 'matlab',
    'c++', 'c/c++', 'perl', 'go', 'typescript', 'bash', 'html', 'css', 'php', 'powershell', 'rust',
    'kotlin', 'ruby', 'dart', 'assembly', 'swift', 'vba', 'lua', 'groovy', 'delphi', 'objective-c', 'objective_c',
    'haskell', 'elixir', 'julia', 'clojure', 'solidity', 'lisp', 'f#', 'fortran', 'erlang', 'apl',
    'cobol', 'ocaml', 'crystal', 'javascript/typescript', 'golang', 'nosql', 'mongodb', 't-sql', 'no-sql',
    'visual basic', 'pascal', 'mongo', 'pl/sql', 'sass', 'vb.net', 'mssql', 'machine learning',
]

keywords_libraries = [
    'scikit-learn', 'jupyter', 'theano', 'openCV', 'spark', 'nltk', 'mlpack', 'chainer', 'fann', 'shogun',
    'dlib', 'mxnet', 'node.js', 'vue', 'vue.js', 'keras', 'ember.js', 'jse/jee',
]

keywords_analyst_tools = [
    'excel', 'tableau', 'word', 'powerpoint', 'looker', 'powerbi', 'outlook', 'azure', 'jira', 'twilio', 'snowflake',
    'shell', 'linux', 'sas', 'sharepoint', 'mysql', 'visio', 'git', 'mssql', 'powerpoints', 'postgresql',
    'spreadsheets',
    'seaborn', 'pandas', 'gdpr', 'spreadsheet', 'alteryx', 'github', 'postgres', 'ssis', 'numpy', 'power_bi',
    'power bi', 'spss', 'ssrs',
    'microstrategy', 'cognos', 'dax', 'matplotlib', 'dplyr', 'tidyr', 'ggplot2', 'plotly', 'esquisse', 'rshiny', 'mlr',
    'docker', 'linux', 'jira', 'hadoop', 'airflow', 'redis', 'graphql', 'sap', 'tensorflow', 'node', 'asp.net', 'unix',
    'jquery', 'pyspark', 'pytorch', 'gitlab', 'selenium', 'splunk', 'bitbucket', 'qlik', 'terminal', 'atlassian',
    'unix/linux',
    'linux/unix', 'ubuntu', 'nuix', 'datarobot',
]

keywords_cloud_tools = [
    'aws', 'azure', 'gcp', 'snowflake', 'redshift', 'bigquery', 'aurora', 'data lake', 'data lakes',
]

keywords_soft_skills = [
    'communication', 'teamwork', 'adaptability', 'flexibility', 'creativity', 'empathy', 'motivation', 'resilience',
    'initiative', 'patience', 'negotiation', 'collaboration', 'accountability', 'integrity', 'multitasking',
    'problem solving', 'time management', 'critical thinking', 'decision making', 'conflict resolution',
    'detail oriented', 'learning agility', 'interpersonal skills', 'emotional intelligence',
    'analytical thinking', 'strategic thinking', 'presentation skills', 'resource management',
]

keywords = keywords_programming + keywords_libraries + keywords_analyst_tools + keywords_cloud_tools + \
    keywords_soft_skills

jobs_all = jobs_all[jobs_all.description.notnull()].reset_index()  # filter out null values

jobs_all['description_tokens'] = ''

for index, row in jobs_all.iterrows():
    # lowercase words
    detail = row.description.lower()
    # tokenize words
    detail = word_tokenize(detail)
    # handle multi-word tokenization (e.g., 'Power BI')
    multi_tokens = [('power', 'bi'), ('data', 'lake'), ('data', 'lakes'), ('machine', 'learning'), ('objective', 'c'),
                    ('visual', 'basic'),
                    ('time', 'management'), ('critical', 'thinking'), ('decision', 'making'),
                    ('conflict', 'resolution'), ('detail', 'oriented'), ('problem', 'solving'), ('learning', 'agility'),
                    ('interpersonal', 'skills'), ('emotional', 'intelligence'), ('analytical', 'thinking'),
                    ('strategic', 'thinking'), ('presentation', 'skills'), ('resource', 'management'),
                    ]
    tokenizer = MWETokenizer(multi_tokens, separator=' ')
    detail = tokenizer.tokenize(detail)
    # remove duplicates
    detail = list(set(detail))
    # filter for keywords only
    detail = [word for word in detail if word in keywords]
    # replace duplicate keywords
    replace_tokens = {'powerbi': 'power bi', 'power_bi': 'power bi',
                      'spreadsheets': 'spreadsheet', 'powerpoints': 'powerpoint'}
    for key, value in replace_tokens.items():
        detail = [d.replace(key, value) for d in detail]
    # add to details list # row.description_tokens = detail
    jobs_all.at[index, 'description_tokens'] = [skill for skill in detail]

### EDA on skills:


def filtered_keywords(jobs_filtered, keywords, title="Keyword Analysis", head=10):
    # get keywords in a column
    count_keywords = pd.DataFrame(jobs_filtered.description_tokens.sum()).value_counts() \
        .rename_axis('keywords').reset_index(name='counts')

    # get frequency of occurrence of word (as word only appears once per line)
    length = len(jobs_filtered)  # number of job postings
    count_keywords['percentage'] = 100 * count_keywords.counts / length

    # plot the results
    count_keywords = count_keywords[count_keywords.keywords.isin(keywords)]
    count_keywords = count_keywords.head(head)
    plt.bar(x="keywords", height="percentage", data=count_keywords,
            color=np.random.rand(len(count_keywords.keywords), 3))
    plt.xlabel("")
    plt.ylabel("Likelihood to be in job posting (%)")
    plt.xticks(rotation=15, ha='right')
    plt.title(title)
    plt.show()
    print(count_keywords)


filtered_keywords(jobs_all, keywords, title="Top Skills for Data Analysts")


### Prepare for SQL:
jobs_all_sql = jobs_all[['title', 'company_name', 'location', 'via', 'date_time',
                         'salary_standardized', 'description_tokens']]
# transform token lists in column into strings
jobs_all_sql['description_tokens'] = jobs_all_sql['description_tokens'].map(lambda x: ', '.join(y for y in x))
# export CSV file
jobs_all_sql.to_csv('jobs_sql.csv', index=False)