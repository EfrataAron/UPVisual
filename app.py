import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("Quick Data Analysis Web App")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display the dataset
        st.subheader("Dataset Preview")
        st.write(df)

        # Basic Statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())

        # Data Visualization: Single Column
        st.subheader("Single Column Visualization")
        single_column = st.selectbox("Select a column to plot (Histogram)", df.columns)
        if df[single_column].dtype in ["float64", "int64"]:
            fig, ax = plt.subplots()
            sns.histplot(df[single_column], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Selected column is not numeric.")

        # Data Visualization: Two Columns
        st.subheader("Two Columns Visualization")
        col1, col2 = st.columns(2)

        with col1:
            column_x = st.selectbox("Select X-axis column", df.columns)

        with col2:
            column_y = st.selectbox("Select Y-axis column", df.columns)

        if column_x and column_y:
            if df[column_x].dtype in ["float64", "int64"] and df[column_y].dtype in [
                "float64",
                "int64",
            ]:
                # Scatter Plot
                st.write("Scatter Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[column_x], y=df[column_y], ax=ax)
                st.pyplot(fig)

                # Line Plot
                if st.checkbox("Show Line Plot"):
                    st.write("Line Plot")
                    fig, ax = plt.subplots()
                    sns.lineplot(x=df[column_x], y=df[column_y], ax=ax)
                    st.pyplot(fig)

                # Bar Plot
                if st.checkbox("Show Bar Plot"):
                    st.write("Bar Plot")
                    fig, ax = plt.subplots()
                    sns.barplot(x=df[column_x], y=df[column_y], ax=ax)
                    st.pyplot(fig)
            else:
                st.write("Both columns must be numeric to plot.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting file upload.")

# Footer
st.write("Made with ❤️ using Streamlit.")
