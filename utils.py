import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_det_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


@st.cache(persist=True)
def load_data():
    df = pd.read_csv('C:/Users/admin/Downloads/PythonTutorials/Web-App/credit_dataset.csv')
    df['FAMILY SIZE'] = df['FAMILY SIZE'].astype(int)
    
    le = LabelEncoder()
    label_df = df.copy()
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    for row in object_cols:
        label_df[row] = le.fit_transform(df[row])
    label_df.drop(label_df.columns[0], axis=1, inplace=True)
    return label_df

@st.cache(persist=True)
def split(df):
    x = df.copy()
    y = x.pop('TARGET')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, class_names, y_pred):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()

    if 'AUC Curve' in metrics_list:
        st.subheader("AUC Curve")
        false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_test, y_pred)
        plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
        st.pyplot()    