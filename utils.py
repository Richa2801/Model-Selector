import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_det_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

@st.cache(suppress_st_warning=True)

@st.cache(persist=True)
def load_data(uploaded_file):
    le_name_mapping = {}
    df = pd.read_csv(uploaded_file)
    df.dropna(inplace=True, axis=0)
    le = LabelEncoder()
    label_df = df.copy()
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)
    if object_cols:
        for row in object_cols:
            label_df[row] = le.fit_transform(df[row])
        
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    label_df.drop(label_df.columns[0], axis=1, inplace=True)
    #label_df.drop(['ID', 'GENDER', 'REALITY','NO_OF_CHILD', 'HOUSE_TYPE', 'FLAG_MOBIL', 'WORK_PHONE', 'E_MAIL'], axis=1, inplace=True)
    return label_df, le_name_mapping


@st.cache(persist=True)
def split(df, answer):
    data = df.values
    x = data[:, :-1]
    y = data[:, -1]
    if answer:
        from imblearn.over_sampling import SMOTE
        x,y = SMOTE().fit_resample(x,y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, y_pred):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test)
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