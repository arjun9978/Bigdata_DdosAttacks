**Big Data DDoS Attack Detection and Classification**
Overview
This project focuses on detecting and classifying DDoS attacks using big data analytics and machine learning models. It leverages the CICDDoS2019 dataset to analyze network traffic patterns and improve cybersecurity measures.

Project Details
Institution: Jaypee Institute of Information Technology, Noida

Supervisor: Dr. Neeraj Jain

Course: Introduction to Big Data and Data Analytics (20B12CS333)

Authors: Arjun Pratap Aggarwal, Daksh Billa

Dataset
Source: CICDDoS2019

Size: 125,170 training samples, 306,201 testing samples

Features: 78 network traffic attributes (timestamps, protocols, IPs, ports, etc.)

Attack Types: NTP, DNS, MSSQL, UDP, SYN, and more

Installation & Setup
Clone the Repository
git clone https://github.com/arjun9978/Bigdata_DdosAttacks.git
cd Bigdata_DdosAttacks
Install Dependencies
pip install -r requirements.txt
Download Dataset
import kagglehub
dataset_path = kagglehub.dataset_download('dhoogla/cicddos2019')
Implementation
Data Processing
Feature Engineering: StandardScaler, MinMaxScaler, RobustScaler

Feature Selection: Variance Threshold, Label Encoding, One-Hot Encoding

Handling Missing Values: Imputation techniques applied

Machine Learning Models Used
Random Forest Classifier

Decision Tree Classifier

Logistic Regression

Model Training & Evaluation
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
predictions = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
Results & Insights
The model successfully detects and classifies DDoS attacks with high accuracy.

Important network attributes such as flow duration and packet size play a crucial role in classification.

Future Improvements
Exploring deep learning models for enhanced detection.

Optimizing feature selection for efficiency.

Developing real-time DDoS detection capabilities.

License
This project is open-source under the MIT License.

Contact
For inquiries and collaboration, reach out via:

GitHub: arjun9978

Email: [your.email@example.com]



