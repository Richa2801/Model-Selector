import streamlit as st
import utils
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, roc_auc_score


st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    st.title("Confused which classifier model is the best for your data?")
    st.sidebar.title("Confused which classifier model is the best for your data?")
    st.subheader("Make your decision here")
    st.sidebar.subheader("Make your decision here")

    try:
        uploaded_file = st.file_uploader("Upload your csv file to begin")
        df, labels = utils.load_data(uploaded_file)
        if labels:
            st.write("Labels:")
            st.text(labels)
        st.success("Great work! Now please make your choices from the sidebar")
        answer = st.sidebar.radio("Is your dataset imbalanced?", ("False","True"))
        cls = st.sidebar.radio("Is your dataset: ", ("Binary","Multiclass"))
        x_train, x_test, y_train, y_test = utils.split(df, answer)

        
        if cls=="Binary":
        
            st.sidebar.subheader("Choose Classifier")
            classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Decision Tree Classifier", "Light GBM"))


            if classifier == 'Light GBM':


                    if st.sidebar.button("Classify", key="classify"):
                        st.subheader("Light GBM results")
                        model = LGBMClassifier(average='weighted')
                        model.fit(x_train, y_train)
                        accuracy = model.score(x_test, y_test)
                        y_pred = model.predict(x_test)                   
                        st.write("Accuracy: ", accuracy.round(2))
                        st.write("Precision: ", precision_score(y_test, y_pred).round(2))
                        st.write("Recall: ", recall_score(y_test, y_pred).round(2))
                        st.write("AUC: ", roc_auc_score(y_test, y_pred))
                        utils.plot_metrics(metrics, model, x_test, y_test, y_pred)


            if classifier == 'Logistic Regression':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
                max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

                metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve","Precision-Recall Curve", "AUC Curve"))

                if st.sidebar.button("Classify", key="classify"):
                    st.subheader("Logistic Regression Results")
                    model = LogisticRegression(C=C, max_iter=max_iter)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred).round(2))
                    st.write("AUC: ", roc_auc_score(y_test, y_pred))
                    utils.plot_metrics(metrics, model, x_test, y_test, y_pred)


            if classifier == 'Decision Tree Classifier':
                st.sidebar.subheader("Model Hyperparameters")
                criteria = st.sidebar.selectbox("Criterion", ("gini","entropy"))
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=2, key='max_depth')
                metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                        "Precision-Recall Curve", "AUC Curve"))

                if st.sidebar.button("Classify", key="classify"):
                    st.subheader("Decision Tree Results")
                    model  = DecisionTreeClassifier(criterion=criteria, random_state=0, max_depth=max_depth) 
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precision: ", precision_score(y_test, y_pred).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred).round(2))
                    st.write("AUC: ", roc_auc_score(y_test, y_pred))
                    utils.plot_metrics(metrics, model, x_test, y_test, y_pred)

        else:
            from sklearn.metrics import classification_report
            st.sidebar.subheader("Choose Classifier")
            classifier = st.sidebar.selectbox("Classifier", ("KNN","Random Forest Classification", "SVM"))


            if classifier == 'KNN':
                
                st.sidebar.subheader("Model Hyperparameters")
                n = st.sidebar.number_input("Max_Leaf")
                m = st.sidebar.selectbox("What metric?", ("euclidean", "minkowski"))
                metrics = st.sidebar.multiselect("What matrix to plot?", ("None","Confusion Matrix"))

                if st.sidebar.button("Classify", key="classify"):
                    st.subheader("K Nearest Classifier results")
                    model= KNeighborsClassifier(n_neighbors=round(n), metric=m, p=2) 
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)                 
                    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
                    utils.plot_metrics(metrics, model, x_test, y_test, y_pred)


            if classifier == 'SVM':
                st.sidebar.subheader("Model Hyperparameters")
                c = st.sidebar.number_input("C (Regularization parameter)", 0.1, 1000.0, step=10.0, key='Lr')
                g = st.sidebar.number_input("gamma", 0.0001, 1.0, step=10.0, key='Lr')
                svc = svm.SVC(kernel='linear', C=c, gamma=g)
                metrics = st.sidebar.multiselect("What matrix to plot?", ("None","Confusion Matrix"))

                if st.sidebar.button("Classify", key="classify"):
                    st.subheader("SVM Results")
                    svc.fit(x_train, y_train)
                    y_pred = svc.predict(x_test)
                    model=svc
                    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
                    utils.plot_metrics(metrics, model, x_test, y_test, y_pred)


            if classifier == 'Random Forest Classification':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("This is the number of trees in the forest", 100, 5000, step=10,
                                                    key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=2, key='max_depth')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
                metrics = st.sidebar.multiselect("What matrix to plot?", ("None","Confusion Matrix"))

                if st.sidebar.button("Classify", key="classify"):
                    st.subheader("Random Forest Results")
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.dataframe(classification_report(y_test, y_pred, output_dict=True))
                    utils.plot_metrics(metrics, model, x_test, y_test, y_pred)            

        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Data Set (Classification)")
            st.write(df)

        profile = ProfileReport(df)
        if st.sidebar.checkbox("Show pandas-profiling report", False):
            st.subheader("Pandas Profiling Report")
            st_profile_report(profile)

    except:
        st.info("We will drop missing values and perform label encoding :)")
    


if __name__ == '__main__':
    main()
