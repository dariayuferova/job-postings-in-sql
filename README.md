# job-postings-sql
Repository for an SQL project analyzing Data Analyst Job Postings in the US.

This project is based on [Data Analyst Job Postings [Pay, Skills, Benefits] dataset](https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search?datasetId=2614070&searchQuery=Data+Analyst+Skill+Analysis) by __Luke Barousse__ and his [analysis](https://www.kaggle.com/code/lukebarousse/data-analyst-skill-analysis/notebook) (see [license](https://www.apache.org/licenses/LICENSE-2.0)).  

In his data collection process Luke uses SerpAPI to pull job postings from Google's search results for Data Analyst positions in the United States. Data collection started on November 4th, 2022, and the results are updated daily - check it out, as well as his [wonderful YouTube channel](https://www.youtube.com/@LukeBarousse)! Thank you Luke for everything you're doing for Data Analytics community! 🙌

When examining Luke's results, I was surprised to see SQL (not Python!) at the very top of the skills for Data Analysts. So it only made sense for me to try and process this data using the employers' favorite tool!

### The Repository's Contents
The purpose of this project is mainly to practice my SQL skills, thus, this repository includes Jupyter Notebook file with step-by-step analysis in SQL.   
The Python code used is, again, mostly [Luke's](https://github.com/lukebarousse) intellectual property, I just tweaked a couple of things and exported a CSV file with a few columns I needed.  
jobs_sql.csv - file I exported using Python code.  
states_abbr_name.csv, states_coord.csv, states_gdp.csv - files I used for visualizations.  
