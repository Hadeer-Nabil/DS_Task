import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, log_loss
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

#helper functions 
def classifier_selection(X_train, X_test, y_train, y_test):
    # Logging for Visual Comparison
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes","LogisticRegression","LDA", "QDA"]

    classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=50),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
    
    for name, classifier in zip(names, classifiers):
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test , y_pred)
        loss = log_loss(y_test, y_pred)
        print(name)
        print(score)
        #print(precision_recall_fscore_support(y_test, y_pred, average=None))
        log_entry = pd.DataFrame([[name, accuracy*100, loss]], columns=log_cols)
        log = log.append(log_entry)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    plt.xlabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

    sns.set_color_codes("muted")
    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

    plt.xlabel('Log Loss')
    plt.title('Classifier Log Loss')
    plt.show() 


def nn_base_classifier():
	# create the Neural Network model
    classifier = Sequential()

    classifier.add(Dense(activation="relu", input_dim=56, units=114, kernel_initializer="normal"))

    classifier.add(Dense(activation="relu", units=114, kernel_initializer="normal"))
        
    classifier.add(Dense(activation="relu", units=114, kernel_initializer="normal"))
    
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="normal"))

    classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

def nn_base_classifier_with_channels():
	# create Neural Network model
    classifier = Sequential()

    classifier.add(Dense(activation="relu", input_dim=233, units=117, kernel_initializer="normal"))

    classifier.add(Dense(activation="relu", units=117, kernel_initializer="normal"))
    
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="normal"))

    classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

def nn_classifier(optimizer,X_train,y_train, X_test, y_test):
    
    classifier = nn_base_classifier()
    print("******")
    print(optimizer)
    history = classifier.fit(X_train,y_train, epochs = 50,validation_data=(X_test, y_test))
    proba = classifier.predict_proba(X_test, batch_size=1)
    print(proba)
    return classifier

def evaluate_nn_classifier(model, X,y):
    
    n_split=10
    for train_index,test_index in KFold(n_split).split(X):
        x_train,x_test=X.iloc[train_index],X.iloc[test_index]
        y_train,y_test=y.iloc[train_index],y.iloc[test_index]

        model.fit(x_train, y_train,epochs=20)
  
        print('Model evaluation ',model.evaluate(x_test,y_test))
    
    
#read the data from the csv file into Dataframe
dataset = pd.read_csv('data_senior_ds_challenge_2019-06-05_11-28.csv')


#checking the text fields if they are categorical or not
"""
dataset['question.travel_behavior'].unique()
dataset['question.employment_status'].unique()
dataset['question.tv_watching_frequency'].unique()
dataset['question.technology_affinity'].unique()
dataset['question.personal_area_of_work'].unique()
dataset['question.monthly_household_income'].unique()
"""

# removing fields that will act like a noise for example meta.wave_label and the dependant variable 
X_with_channels = dataset.drop(['question.brand_awareness_client', 'meta.uuid','meta.wave_label','dem.country_code','question.us_cities','question.uk_cities'] , axis = 1)

# for mapping True to 1 and False to 0
X_with_channels = X_with_channels*1

y_with_channels= dataset.loc[:,['question.brand_awareness_client']]

#selecting the columns without the channels columns (facebook channel flag and etc.)
dataset_without_channels = dataset.iloc[:,2:19]
#dataset_without_channels.head()

#sns.pairplot(dataset_without_channels, hue = 'question.brand_awareness_client')

#removing fields that is mainly empty 
dataset_without_channels.drop(['dem.country_code','question.us_cities','question.uk_cities'],axis=1,inplace=True)



# selecting columns 
X = dataset_without_channels.loc[:,['dem.age', 'dem.education_level', 'question.tv_watching_frequency',
       'question.technology_affinity', 'question.travel_behavior',
       'question.de_cities', 'question.employment_status',
       'question.work_start_up', 'question.personal_area_of_work',
       'question.monthly_household_income']]
y = dataset_without_channels.loc[:,['question.brand_awareness_client']]

# defining columns that contains categorical data 
categorical_col = ['dem.education_level', 'question.tv_watching_frequency',
       'question.technology_affinity', 'question.travel_behavior',
       'question.de_cities', 'question.employment_status',
       'question.work_start_up', 'question.personal_area_of_work',
       'question.monthly_household_income']

# making columns categorical (Type)
for col in categorical_col:
    X[col] = pd.Categorical(X[col])
    X_with_channels[col] = pd.Categorical(X_with_channels[col])
    dataset_without_channels[col] = pd.Categorical(dataset_without_channels[col])


# performing one hot encoder on the categorical columns
X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
X_with_channels = pd.get_dummies(X_with_channels, prefix_sep='_', drop_first=True)
dataset_without_channels = pd.get_dummies(dataset_without_channels, prefix_sep='_', drop_first=True)

#sns.pairplot(dataset_without_channels, hue = 'question.brand_awareness_client')

# maping True and False to 1 and 0
y['question.brand_awareness_client'] = y['question.brand_awareness_client'].astype(int)

# spliting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

from sklearn.model_selection import train_test_split
X_train_with_channels, X_test_with_channels, y_train_with_channels, y_test_with_channels = train_test_split(X_with_channels, y, test_size = 0.25, random_state=0)


optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
   
#checking which best optimizer for the Neural Network  
#for opt in optimizer: 
 #   classifier = nn_classifier(opt,X_train,y_train, X_test, y_test)

#classifier_selection(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())


# Keras classifier will be used since it scored the best Accuracy so far (91%-95%) 
#with the evaluation with k-fold cross validation
classifier = nn_classifier('Adamax',X_train,y_train, X_test, y_test)  
y_prob = classifier.predict(X_test) 

classifier = nn_base_classifier()
evaluate_nn_classifier(classifier,X,y)
