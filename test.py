import streamlit as st
import numpy as np
import utils
import lightgbm as lgb

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import precision_score, recall_score, roc_auc_score

import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    st.title("Credit Card Fraud Detection Web App")
    st.sidebar.title("Credit Card Fraud Detection Web App")
    st.markdown("Is the customer genuine or fraud?")
    st.sidebar.markdown("Is the customer genuine or fraud?")


    df = utils.load_data()
    x_train, x_test, y_train, y_test = utils.split(df)
    class_names = ["Genuine", "Fraud"]

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')
    label = ["Genuine", "Fraud"]
    nums = [df.TARGET.value_counts()[0], df.TARGET.value_counts()[1]]
    ax.pie(nums, labels = label, autopct='%1.2f%%')
    st.pyplot() 

    if st.sidebar.radio("Is the dataset imbalanced?", ("False", "True")):
        import imblearn
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        # define pipeline
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)

        x_train, y_train = pipeline.fit_resample(x_train, y_train)
        x_test, y_test = pipeline.fit_resample(x_test, y_test)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression",
                                                     "Random Forest Classification", "Light GBM"))


    if classifier == 'Light GBM':
            st.sidebar.subheader("Model Hyperparameters")
            n = st.sidebar.number_input("Max_Leaf", 31, 100, step=1)
            max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')
            metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                    "Precision-Recall Curve", "AUC Curve"))

            if st.sidebar.button("Classify", key="classify"):
                st.subheader("Light GBM results")
                model = LGBMClassifier(max_leaf=n, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                st.write("AUC: ", roc_auc_score(y_test, y_pred))
                utils.plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve", "AUC Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("AUC: ", roc_auc_score(y_test, y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names,y_pred)


    if classifier == 'Random Forest Classification':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("This is the number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=2, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve", "AUC Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            st.write("AUC: ", roc_auc_score(y_test, y_pred))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names, y_pred)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Credit Card Fraud Data Set (Classification)")
        st.write(df)

    profile = ProfileReport(df)
    if st.sidebar.checkbox("Show pandas-profiling report", False):
        st.subheader("Pandas Profiling Report")
        st_profile_report(profile)
    


if __name__ == '__main__':
    main()