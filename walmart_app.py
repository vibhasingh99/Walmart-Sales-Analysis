import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Walmart Sales Dashboard")

df = pd.read_csv("cleaned_walmart_data.csv")

if st.checkbox("Show raw data"):
    st.dataframe(df.head())

st.subheader("Gross Income by Product Line")
fig1, ax1 = plt.subplots()
sns.barplot(data=df, x="Product line", y="gross income", estimator=sum, ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("City-wise Total Sales")
city_sales = df.groupby('City')['Total'].sum()
fig, ax = plt.subplots()
ax.pie(city_sales, labels=city_sales.index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  
st.pyplot(fig)

st.subheader("Gender vs Payment Method")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="Payment", hue="Gender", ax=ax3)
st.pyplot(fig3)


st.subheader("Month-wise Total Sales")
df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month
monthly_sales = df.groupby('Month')['Total'].sum().reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sales['Month'] = monthly_sales['Month'].apply(lambda x: month_names[x - 1])
monthly_sales['Month'] = pd.Categorical(monthly_sales['Month'], categories=month_names, ordered=True)
monthly_sales = monthly_sales.sort_values('Month')
fig, ax = plt.subplots()
sns.lineplot(data=monthly_sales, x='Month', y='Total', marker='o', ax=ax)
st.pyplot(fig)

st.subheader("Average Rating by Product Line")
avg_rating = df.groupby('Product line')['Rating'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=avg_rating, x='Product line', y='Rating', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Machine Learning Section 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month

X = df[["Gender", "City", "Product line", "Payment", "Total", "Hour", "Month"]].copy()
y = df["Customer type"]

le = LabelEncoder()
for col in X.select_dtypes(include='object'):
    X[col] = le.fit_transform(X[col])
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.markdown(f"**Accuracy:** `{accuracy:.2f}`")

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
st.pyplot(fig_cm)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(2))
