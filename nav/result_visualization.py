import streamlit as st
import pandas as pd
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns

def show(models, metrics, le, X):
    st.title("Results and Visualization")

    st.subheader("Model Metrics")
    metrics_df = pd.DataFrame(metrics).transpose()
    st.write(metrics_df)

    st.subheader("Decision Tree Visualization")
    if st.checkbox("Plot Decision Tree"):
        model = models['Decision Tree']
        
        dot_data = export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=X.columns, class_names=le.classes_
        )
        
        graph = pydotplus.graph_from_dot_data(dot_data)
        image = graph.create_png()
        st.image(image, use_column_width=True)

    st.subheader("Model Comparison")
    plot_metrics_comparison(metrics_df)

def plot_metrics_comparison(metrics_df):
    metrics_df['accuracy'] = metrics_df['accuracy'] * 100
    metrics_df['precision'] = metrics_df['precision'] * 100

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    sns.barplot(x=metrics_df.index, y=metrics_df['accuracy'], ax=axs[0])
    axs[0].set_title("Accuracy Comparison")
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("Accuracy (%)")

    sns.barplot(x=metrics_df.index, y=metrics_df['precision'], ax=axs[1])
    axs[1].set_title("Precision Comparison")
    axs[1].set_ylim(0, 100)
    axs[1].set_ylabel("Precision (%)")

    st.pyplot(fig)

    # Additional graphs
    st.subheader("Additional Visualizations")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(metrics_df[['accuracy', 'precision']], annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Heatmap of Model Metrics")
    st.pyplot(fig)
