import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# App Titles
st.title("UPVisual Advanced Data Analysis, Cleaning, and Export Web App")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Display the dataset
        st.subheader("Dataset Preview")
        st.write(df)

        # Basic Statistics
        st.subheader("Basic Statistics")
        st.write(df.describe())

        # Data Cleaning Options
        st.subheader("Data Cleaning")

        # Handle Missing Values
        if df.isnull().values.any():
            st.write("Columns with Missing Values:")
            st.write(df.isnull().sum())
            clean_option = st.radio(
                "How do you want to handle missing values?",
                ("Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode")
            )
            if clean_option == "Drop rows with missing values":
                df = df.dropna()
            elif clean_option == "Fill with mean":
                df = df.fillna(df.mean())
            elif clean_option == "Fill with median":
                df = df.fillna(df.median())
            elif clean_option == "Fill with mode":
                df = df.fillna(df.mode().iloc[0])
            st.write("Updated Dataset:")
            st.write(df)

        # Remove Duplicates
        if st.checkbox("Remove Duplicate Rows"):
            df = df.drop_duplicates()
            st.write("Duplicates removed. Updated dataset:")
            st.write(df)

        # Outlier Detection and Removal
        st.subheader("Outlier Detection")
        if st.checkbox("Remove Outliers (Using IQR)"):
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            st.write("Outliers removed. Updated dataset:")
            st.write(df)

        # Column Renaming
        st.subheader("Rename Columns")
        column_to_rename = st.selectbox("Select a column to rename", df.columns)
        new_column_name = st.text_input("Enter the new column name", column_to_rename)
        if st.button("Rename Column"):
            df = df.rename(columns={column_to_rename: new_column_name})
            st.write("Column renamed successfully. Updated dataset:")
            st.write(df)

        # Data Scaling
        st.subheader("Data Scaling")
        scale_option = st.radio("Choose scaling method", ("None", "Standardization (Z-score)", "Normalization (Min-Max)"))
        if scale_option != "None":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler() if scale_option == "Standardization (Z-score)" else MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            st.write(f"Data scaled using {scale_option}. Updated dataset:")
            st.write(df)

        # Data Visualization
        st.subheader("Data Visualization")

        # Single Column Visualization
        single_column = st.selectbox("Select a column for histogram", df.columns)
        if df[single_column].dtype in ['float64', 'int64']:
            fig, ax = plt.subplots()
            sns.histplot(df[single_column], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Selected column is not numeric.")

        # Two Columns Visualization
        col1, col2 = st.columns(2)
        with col1:
            column_x = st.selectbox("Select X-axis column", df.columns)
        with col2:
            column_y = st.selectbox("Select Y-axis column", df.columns)

        if column_x and column_y:
            if df[column_x].dtype in ['float64', 'int64'] and df[column_y].dtype in ['float64', 'int64']:
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

        # Export Cleaned Data
        st.subheader("Export Cleaned Data")
        file_format = st.radio("Choose file format for export", ("CSV", "Excel"))
        if file_format == "CSV":
            cleaned_file = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=cleaned_file,
                file_name="cleaned_data.csv",
                mime="text/csv",
            )
        elif file_format == "Excel":
            cleaned_file = df.to_excel(index=False, engine='openpyxl')
            st.download_button(
                label="Download Cleaned Data as Excel",
                data=cleaned_file,
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting file upload.")
    

st.subheader("Correlation Analysis")
if st.checkbox("Show Correlation Heatmap"):
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)



st.subheader("Machine Learning Model")
if st.checkbox("Train a Model"):
    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Features", [col for col in df.columns if col != target])
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    if target and features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")

st.subheader("Data Transformation")
transformation = st.radio("Select transformation", ["None", "Log Transformation", "Square Root Transformation"])
if transformation != "None":
    selected_column = st.selectbox("Select column for transformation", df.columns)
    if transformation == "Log Transformation":
        df[selected_column] = np.log1p(df[selected_column])
    elif transformation == "Square Root Transformation":
        df[selected_column] = np.sqrt(df[selected_column])
    st.write("Transformed Dataset:")
    st.write(df)

st.subheader("Time Series Analysis")
time_column = st.selectbox("Select Time Column", df.columns)
value_column = st.selectbox("Select Value Column", [col for col in df.columns if col != time_column])
if pd.api.types.is_datetime64_any_dtype(df[time_column]):
    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    fig, ax = plt.subplots()
    df[value_column].plot(ax=ax)
    st.pyplot(fig)

from sklearn.cluster import KMeans

st.subheader("Clustering")
if st.checkbox("Perform Clustering"):
    num_clusters = st.slider("Number of Clusters", 2, 10)
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(numeric_cols)
    df['Cluster'] = clusters
    st.write("Clustered Dataset:")
    st.write(df)

from wordcloud import WordCloud

st.subheader("Text Data Analysis")
text_column = st.selectbox("Select a Text Column", df.columns)
if text_column:
    text_data = " ".join(df[text_column].astype(str))
    wordcloud = WordCloud().generate(text_data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

import folium
from streamlit_folium import st_folium

st.subheader("Geospatial Analysis")
if "Latitude" in df.columns and "Longitude" in df.columns:
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=10)
    for _, row in df.iterrows():
        folium.Marker(location=[row["Latitude"], row["Longitude"]]).add_to(m)
    st_folium(m, width=700, height=500)


# Footer
st.write("Made with ❤️ using Streamlit.")
