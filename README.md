# Explainable-Binary-Classification
An explainable binary classification project to clean data, vectorize data, K-Fold cross validate and apply a model. The model is made explainable by using LIME Explainers.
### <a href='binary-classification-lime.ipynb'>Check the Notebook here...</a>

### Techniques Used

**Part A:**
- Data formatting, schema validation, data merging, concatenation and key-column identification
- Label Correction
- Text Data Cleaning
- K-Fold Cross Validation (5 fold splitting)

**Part B:**
- Defined Data Vectorization Functions
  - Tf-Idf
  - Simple Avg. FastText Sentence Vectors
  - Tf-Idf Weighted FastText Sentence Vectors
- Defined Models
  - Random Forest Classifier (n_estimators=10)
  - Logistic Regression
- Defined LIME Explainer

**Part C:**
- Performed classification on various combinations of classification models (Random Forest & Logistic Regression) and feature sets. The classification model can be switched by changing a flag in Part C Model Configurations.
- Experiment 0 : Training a Word2Vec Model using the given data corpus
- Experiment 1 : Tf-Idf Vectorization + Classification + LIME Explanation
- Experiment 2 : Simple Avg. FastText Sentence Vectors + Classification
- Experiment 3 : Tf-Idf Weighted FastText Sentence Vectors + Classification


### Conclusion
#### Feature Selection
- Since we have very less amount of data, we cannot apply a customized and trained Word2Vec implementation.
- The dataset works the best with Tf-Idf vectorization.
#### Model
- Random Forest is a better model as it is able to give high accuracy results for a variety of feature vectors
- Logistic Regression gave us more explainable results as compared to Random Forest.
#### LIME Explainers
- To avoid over-fitting the model with the specific words of this dataset, we must take care while selecting the words for removal.
- The frequently used words (with low IDF score) are safer to remove as compared to rarely used words (with high IDF score). Thus, an IDF score based threshold can be used to select the words for removal and improve the model accuracy.
