# # Python Data Mining code implementing Naive Bayes, Decision Tree and K-Nearest Neighbours (KNN) Big Data Analysis Algorithms

<b>Note: Please view the 'DOC' pdf file for an in-detail and comprehensive documentation of this project. Contains detailed explanation, its working and visualized results of the code. The program and dataset are added in the 'DataMiningCode' folder of this repository</b>

<h2>Introduction</h2>

<p>
  This program implements a machine learning pipeline to classify data from the SaYoPillow.csv dataset. The program compares the performance of three machine learning models—Naive Bayes, Decision Trees, and K-Nearest Neighbors (KNN)—on a given dataset, determining which model performs best for classification. The dataset used is SaYoPillow.csv, which appears to contain physiological and sleep-related data, with a target variable that is classified into different categories. The performance metrics like accuracy, precision, recall and F1 scores are calculated manually to better demonstrate the understanding and functionality of each metrics. 
</p>
<p>
  Comparing the performance of different models for a given dataset is essential because no single model works best for all types of data. Different models have varying strengths and weaknesses depending on the data's characteristics, such as its distribution, the presence of noise, or class imbalance. This program demonstrates classification of the dataset using three models. However, based on the requirements, more models can be implemented, trained and tested using python libraries and classes and choose better models for classification. Python provides efficient and powerful libraries and classes, primarily within the scikit-learn library. 
</p>

<h2>Python Libraries and Classes used:</h2> 

<ol>
  <li>
    <h3>Naive Bayes</h3>
    <ul>
      <li><b>Library:</b> sklearn.naive_bayes</li>
      <li><b>Class:</b> GaussianNB</li>
      <li><b>Description:</b> The GaussianNB class is used for implementing the Gaussian Naive Bayes algorithm, which assumes that the features follow a normal distribution. It is ideal for small datasets and handles continuous data well</li>
    </ul>
  </li>

  <li>
    <h3>Decision Trees</h3>
    <ul>
      <li><b>Library:</b> sklearn.tree</li>
      <li><b>Class:</b> DecisionTreeClassifier</li>
      <li><b>Description:</b> The DecisionTreeClassifier class builds a decision tree based on the features of the dataset. It uses criteria like "entropy" or "gini" to split nodes. Decision trees are easy to interpret and can handle both numerical and categorical data.</li>
    </ul>
  </li>
  
  <li>
    <h3>K-Nearest Neighbors (KNN)</h3>
    <ul>
      <li><b>Library:</b> sklearn.neighbors</li>
      <li><b>Class:</b> KNeighborsClassifier</li>
      <li><b>Description:</b> The KNeighborsClassifier class implements the KNN algorithm, where a sample is classified based on the majority vote of its nearest neighbors. KNN is simple and effective for smaller datasets, especially where the decision boundary is not linear.</li>
    </ul>
  </li>
</ol>

<h2>
  Brief Step-by-Step Explanation
</h2>

<ol>
  <li><b>Import Libraries:</b></li>
  The program imports several libraries for data manipulation (pandas, numpy), machine learning (scikit-learn, imblearn), and visualization (matplotlib, mlxtend).

  </br>

  <li><b>Load Dataset:</b></li>
  The dataset is loaded into a Pandas DataFrame, and the first few rows are displayed.

  </br>

  <li>
    <b>Data Preprocessing:</b>
    <ul>
      <li><b>Feature Scaling:</b></li>
      The features (independent variables) are scaled to a range of 0 to 1 using MinMaxScaler.
      <li><b>Handling Imbalanced Data:</b></li>
      SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the dataset, generating synthetic samples for the minority class.
    </ul>
  </li>

  </br>

  <li><b>Train-Test Split:</b></li>
  The dataset is split into training and testing sets with an 80-20 ratio.

  </br>

  <li><b>Feature Scaling (Standardization):</b></li>
  The features are standardized using StandardScaler to have a mean of 0 and a standard deviation of 1.

  </br>

  <li>
    <b>Model Training and Evaluation:</b>
    <ul>
      <li>Naive Bayes Classifier:
        <ul>
          <li>The GaussianNB classifier is trained on the standardized data.</li>
          <li>Predictions are made on the test set.</li>
          <li>The confusion matrix and classification report are generated and visualized.</li>
          <li>Accuracy, precision, recall, and F1 score are manually calculated.</li>
        </ul>
      </li>
    </ul>
    <ul>
      <li>Decision Tree Classifier:
        <ul>
          <li>A Decision Tree with entropy as the criterion is trained and evaluated similarly.</li>
        </ul>
      </li>
    </ul>
    <ul>
      <li>K-Nearest Neighbors Classifier:
        <ul>
          <li>A KNN classifier with 5 neighbors is trained and evaluated similarly.</li>
        </ul>
      </li>
    </ul>

  </li>

  <li>
    <b>Comparison of Models:</b>
    <ul>
      <li>Bar charts are created to compare the accuracy, precision, recall, and F1 score of the three models. (For more details, view the 'Doc' pdf within this repository 
      </li>
    </ul>
  </li>
</ol>

<h2>
  Summary
</h2>
<p>
  This program provides a comprehensive approach to classification, addressing class imbalance, standardizing features, and evaluating multiple models using key performance metrics. It concludes with a visual comparison of model performance, which aids in selecting the most suitable model for the given dataset.
</p>
