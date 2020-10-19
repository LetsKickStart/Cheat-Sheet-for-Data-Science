# Cheat-Sheet-for-Data-Science
Hello Everyone! Here is a Cheat Sheet attached for importing, Exploring, selecting, cleaning,filtering,sorting,grouping,Joining, combining, , writing data with additional Resources to start learning Data Science and Machine learning. Hope this would help

<a href = "https://elitedatascience.com/python-cheat-sheet"><span style ="color:blue";>Click Here for Cheat Sheet on Data Science </span></a>


<b>Importing DataPython</b>

<p>Any kind of data analysis starts with getting hold of some data. Pandas gives you plenty of options for getting data into your Python workbook:</p>

<br>
<div>
    <p>pd.read_csv(filename) <span style = "Color: red";># From a CSV file</span></p>
    <p>pd.read_table(filename) <span style = "Color: red";># From a delimited text file (like TSV)</span></span></p>
    <p>pd.read_excel(filename) <span style = "Color: red";># From an Excel file</span></p>
    <p>pd.read_sql(query, connection_object)<span style = "Color: red";> # Reads from a SQL table/database</span></p>
    <p>pd.read_json(json_string) <span style = "Color: red";># Reads from a JSON formatted string, URL or file.</span></p>
    <p>pd.read_html(url)<span style = "Color: red";> # Parses an html URL, string or file and extracts tables to a list of dataframes.</span></p>
    <p>pd.read_clipboard()<span style = "Color: red";> # Takes the contents of your clipboard and passes it to read_table()</span></p>
    <p>pd.DataFrame(dict) <span style = "Color: red";># From a dict, keys for columns names, values for data as lists</span></p>

</div>

<b>Exploring DataPython </b>
<p>Once you have imported your data into a Pandas dataframe, you can use these methods to get a sense of what the data looks like:</p>
<br>
<div>

<p>df.shape()<span style = "Color: red";># Prints number of rows and columns in dataframe</span></p>
<p>df.head(n)<span style = "Color: red";># Prints first n rows of the DataFrame</span></p>
<p>df.tail(n) <span style = "Color: red";># Prints last n rows of the DataFrame</span></span></p>
<p>df.info()<span style = "Color: red";># Index, Datatype and Memory information</span></p>
<p>df.describe()<span style = "Color: red";> # Summary statistics for numerical columns</span></span></p>
<p>s.value_counts(dropna=False)<span style = "Color: red";> # Views unique values and counts</span></p>
<p>df.apply(pd.Series.value_counts) <span style = "Color: red";># Unique values and counts for all columns</span></p>
<p>df.describe() <span style = "Color: red";># Summary statistics for numerical columns</span></p>
<p>df.mean()<span style = "Color: red";> # Returns the mean of all columns</span></p>
<p>df.corr()<span style = "Color: red";> # Returns the correlation between columns in a DataFrame</span></p>
<p>df.count()<span style = "Color: red";> # Returns the number of non-null values in each DataFrame column</span></p>
<p>df.max() <span style = "Color: red";><span style = "Color: red";># Returns the highest value in each column</span></p>
<p>df.min()<span style = "Color: red";> # Returns the lowest value in each column</span></p>
<p>df.median() <span style = "Color: red";># Returns the median of each column</span></p>
<p>df.std()<span style = "Color: red";> # Returns the standard deviation of each column</span></p>

</div>

<b>Selecting DataPython</b>
<p>Often, you might need to select a single element or a certain subset of the data to inspect it or perform further analysis. These methods will come in handy:</p>
<br>
<div>
<p>df[col] <span style = "Color: red";># Returns column with label col as Series</span></p>
<p>df[[col1, col2]]<span style = "Color: red";> # Returns Columns as a new DataFrame</span></p>
<p>s.iloc[0] <span style = "Color: red";># Selection by position (selects first element)</span></p>
<p>s.loc[0]<span style = "Color: red";> # Selection by index (selects element at index 0)</span></p>
<p>df.iloc[0,:] <span style = "Color: red";># First row</span></p>
<p>df.iloc[0,0] <span style = "Color: red";># First element of first column</span></p>
</div>

<b>Data CleaningPython</b>
<p>If you’re working with real world data, chances are you’ll need to clean it up. These are some helpful methods:</p>
<br>
<div>
<p>df.columns = ['a','b','c']<span style = "Color: red";> # Renames columns</span></p>
<p>pd.isnull()<span style = "Color: red";> # Checks for null Values, Returns Boolean Array</span></p>
<p>pd.notnull()<span style = "Color: red";> # Opposite of s.isnull()</span></p>
<p>df.dropna() <span style = "Color: red";># Drops all rows that contain null valudf.dropna(axis=1) # Drops all columns that contain null values</span></p>
<p>df.dropna(axis=1,thresh=n) <span style = "Color: red";># Drops all rows have have less than n non null value</span></p>
<p>df.fillna(x) <span style = "Color: red";># Replaces all null values with x</span></p>
<p>s.fillna(s.mean())<span style = "Color: red";> # Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section</span>)</p>
<p>s.astype(float) <span style = "Color: red";># Converts the datatype of the series to float</span></p>
<p>s.replace(1,'one') <span style = "Color: red";># Replaces all values equal to 1 with 'one'</p>
<p>s.replace([1,3],['one','three'])<span style = "Color: red";> # Replaces all 1 with 'one' and 3 with 'three'</span></p>
<p>df.rename(columns=lambda x: x + 1) <span style = "Color: red";># Mass renaming of columns</span></p>
<p>df.rename(columns={'old_name': 'new_ name'})<span style = "Color: red";> # Selective renaming</span></p>
<p>df.set_index('column_one') <span style = "Color: red";># Changes the index</span></p>
<p>df.rename(index=lambda x: x + 1)<span style = "Color: red";> # Mass renaming of index</span></p>
</div>

<b>filtering, sorting and grouping your data</b>
<p>Methods for filtering, sorting and grouping your data:</p>
<div>
<p>df1.append(df2) <span style = "Color: red";># Adds the rows in df1 to the end of df2 (columns should be identical)</span></span></p>
<p>pd.concat([df1, df2],axis=1) <span style = "Color: red";># Adds the columns in df1 to the end of df2 (rows should be identical)</span></p>
<p>df1.join(df2,on=col1,how='inner') <span style = "Color: red";># SQL-style joins the columns in df1 with the columns on df2 where the rows for col have identical values. how can be one of 'left', 'right', 'outer', 'inner'<strong> </strong></span></p>
</div>

<b>Writing DataPython</b>
<br>

<div>
<p>df.to_csv(filename)<span style = "Color: red";> # Writes to a CSV file</span></p>
<p>df.to_excel(filename) <span style = "Color: red";># Writes to an Excel file</span></p>
<p>df.to_sql(table_name, connection_object)<span style = "Color: red";> # Writes to a SQL table</span></p>
<p>df.to_json(filename)<span style = "Color: red";> # Writes to a file in JSON format</p>
<p>df.to_html(filename) <span style = "Color: red";><span style = "Color: red";># Saves as an HTML table</span></p>
<p>df.to_clipboard()<span style = "Color: red";> # Writes to the clipboard</span></p>
 </div>
 
 

<h2>Overview of Machine Learning:</h2>

The Scikit-Learn library contains useful methods for training and applying machine learning models. Our <a href="https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn"><b>Scikit-Learn tutorial</b></a> provides more context for the code below.

For a complete list of the Supervised Learning, Unsupervised Learning, and Dataset Transformation, and Model Evaluation modules in Scikit-Learn, please refer to its <a href = "https://scikit-learn.org/stable/user_guide.html"><b>user guide</b></a>.

<u><strong>Code<strong></u>

<b>Import libraries and modules</b>
import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 
 
<b>Load red wine data.</b>
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
 
<b>Split data into training and test sets</b>
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
 
<b>Declare data preprocessing steps</b>
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
 
<b>Declare hyperparameters to tune</b>
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
<b>Tune model using cross-validation pipeline</b>
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
clf.fit(X_train, y_train)
 
<b>Refit on the entire training set</b>
<b> No additional code needed if clf.refit == True (default is True)</b>
 
<b>Evaluate model pipeline on test data</b>
pred = clf.predict(X_test)
print r2_score(y_test, pred)
print mean_squared_error(y_test, pred)
 
<b>Save model for future use</b>
joblib.dump(clf, 'rf_regressor.pkl')
<b>To load: clf2 = joblib.load('rf_regressor.pkl')</b>


<h4>Conclusion<h4>
 <br>
<p>We’ve barely scratching the surface in terms of what you can do with Python and data science, but we hope this cheatsheet has given you a taste of what you can do!</p>

<p>This post was kindly provided by our friend Kara Tan. Kara is a cofounder of <a href = "https://www.altitudelabs.com/"><b> Altitude Labs</b></a>, a full-service app design and development agency that specializes in data driven design and personalization.</p>


<strong>Additional Resources:<strong>

<ul>

<li><a href = "https://elitedatascience.com/machine-learning-projects-for-beginners"><b>8 Fun Machine Learning Projects for Beginners</b></a></li>

<li><a href = "https://elitedatascience.com/datasets"><b> Datasets for Data Science and Machine Learning</b></a></li>

<li><a href = "https://elitedatascience.com/beginner-mistakes"><b>9 Mistakes to Avoid When Starting Your Career in Data Science</b></a></li>
<li><a href = "https://elitedatascience.com/data-science-resources"><b> Free Data Science Resources for Beginners</b></a></li>
<li><a href = "https://elitedatascience.com/machine-learning-algorithms"><b>Overview of Modern Machine Learning Algorithms</b></a></li>
</ul>
