
📝 Project Description

This project focuses on analyzing Zomato restaurant data across various cities in India. It aims to extract meaningful insights into the Indian food industry by performing sentiment analysis on customer reviews and clustering restaurants into distinct segments. The core objective is to assist customers in finding the best restaurants in their locality and enable Zomato to identify areas for growth and improvement, leveraging valuable information around cuisine, costing, and user reviews.

💡 Problem Statement

The rapid growth of the restaurant business in India and the increasing reliance on platforms like Zomato necessitate a deeper understanding of their vast restaurant data. The primary problem addressed is how to extract actionable insights from Zomato's restaurant and review data to benefit both customers and the company. Specifically, this involves:

Analyzing customer review sentiments to understand user perceptions and identify areas of strength and weakness for restaurants.

Effectively clustering Zomato restaurants into distinct segments based on their characteristics (e.g., cuisine, cost, reviews) to help customers easily discover restaurants best suited to their preferences and locality.

Providing actionable business insights to Zomato, helping them identify market trends, areas for operational improvement, and opportunities for strategic growth within the Indian food industry.

🚀 Key Steps & Methodology

This project followed a comprehensive machine learning pipeline:

Data Understanding and Wrangling:

Initial exploration of Restaurant (105 entries, 6 columns) and Review (10,000 entries, 7 columns) datasets.

Addressed crucial data quality issues: type conversions for Cost and Rating, removal of 36 duplicate review rows, handling of 54 missing Collections values, and extracting Reviewer_Reviews and Reviewer_Followers from metadata.

All data was cleaned and transformed into numerical formats.

Exploratory Data Analysis (EDA) and Visualization:

Analyzed restaurant cost distribution (mostly budget-friendly, right-skewed).

Examined customer rating distribution (overwhelmingly positive, indicating high satisfaction).

Identified top cuisines by popularity (North Indian, Chinese).

Compared cuisine popularity with average ratings, revealing a quality gap in some popular cuisines.

Investigated reviewer activity and influence, noting a few highly prolific users.

Observed significant growth in review activity from mid-2018 but a slight decline in average ratings during growth.

Identified weak correlations between rating and reviewer metrics, but moderate correlations among reviewer activity metrics.

Feature Engineering and Preprocessing:

Outlier Treatment: Applied capping at the 99th percentile to skewed numerical features (Cost, Pictures, Reviewer_Reviews, Reviewer_Followers) to reduce undue influence.

Categorical Encoding: Used One-Hot Encoding (str.get_dummies(), pd.get_dummies(), astype(int)) for multi-valued and nominal categorical features.

Textual Data Preprocessing: Implemented a comprehensive NLP pipeline including contraction expansion, lowercasing, punctuation/URL/digit removal, stopwords removal, tokenization, POS-aware lemmatization, and TF-IDF Vectorization, transforming review text into 3441 numerical features.

Feature Manipulation: Aggregated review-level numerical and TF-IDF features to the restaurant level, combining them with original restaurant features into a final_clustering_df of (100, 3650) dimensions.

Dimensionality Reduction: Applied Principal Component Analysis (PCA) to reduce features from 3650 to 50 principal components, retaining ~67% of total variance.

Data Scaling: Utilized StandardScaler on the PCA components to ensure features are on a comparable scale for clustering.

📊 ML Model Implementation & Selection

Three unsupervised clustering models were implemented and evaluated:

K-Means (K=4):

Performance: Inertia: 4725.57, Silhouette: 0.179

Cluster Distribution: Highly imbalanced (97, 1, 1, 1)

Agglomerative Clustering (K=4):

Performance: Inertia: 4705.92, Silhouette: 0.199

Cluster Distribution: Slightly better than K-Means, but still skewed (91, 5, 3, 1)

Gaussian Mixture Models (GMM) (K=3):

Performance: AIC: 3375.44, BIC: 13736.21, Silhouette: 0.083

Cluster Distribution: Significantly more balanced (54, 37, 9)

✨ Chosen Model: Gaussian Mixture Model (GMM) with K=3

Justification: GMM was selected as the final model due to its superior ability to produce a more balanced and actionable cluster distribution. While its Silhouette Score was lower, its flexibility in modeling complex cluster shapes and yielding segments of actionable sizes (e.g., 37 or 9 restaurants vs. singletons) is paramount for deriving meaningful business strategies.

💡 Cluster Characterization (Feature Importance)

The 3 GMM clusters were characterized by analyzing the mean of scaled original features to understand what defines each segment:

Cluster 0 (Popular & High-Engagement): Higher cost, above-average rating, high pictures/reviewer engagement, often late-night. Represents well-regarded establishments.

Cluster 1 (Premium, High-Visibility, Mixed-Review): Highest cost, highest pictures/reviewer engagement, but below-average rating. Potentially trendy but inconsistent.

Cluster 2 (Budget-Friendly, Standard Quality): Lower cost, slightly below-average rating, lower pictures/reviewer engagement. Represents the broad base of affordable restaurants with varied experiences.

📈 Business Impact & Conclusion

This project successfully demonstrates how Zomato can leverage its data for strategic segmentation. The identified clusters provide a clear framework for:

Targeted Marketing: Developing specific campaigns for each distinct restaurant segment.

Product Development: Curating specialized restaurant collections for customers.

Operational Improvements: Identifying areas for enhancement within specific segments.

Competitive Analysis: Understanding the market composition.

The GMM's balanced cluster output offers the most practical and interpretable segmentation, making it a powerful tool for Zomato to enhance user experience, optimize business strategies, and drive growth in the dynamic Indian food industry.

⚙️ How to Run the Notebook

Clone the Repository:
git clone -- https://github.com/amantiwari-java/Zomato-Restaurant-Clustering-Project

Open in Google Colab: Upload the Zomato_Restaurant_Clustering_Project_Aman.ipynb file to Google Colab.

Upload Data: Upload Zomato Restaurant names and Metadata.csv and Zomato Restaurant reviews.csv to your Colab session storage.

Run All Cells: Go to Runtime > Run all to execute the entire analysis and model training pipeline.

🔮 Future Work (Optional)

Model Deployment: The trained GMM model has been saved (gmm_restaurant_clustering_model.joblib) for potential deployment on a live server to predict clusters for new restaurants.

Sentiment Analysis: Further develop the sentiment analysis aspect of review data to provide a more nuanced understanding of customer feedback within each cluster.

Time-Series Forecasting: Analyze review trends to forecast future restaurant popularity or market shifts.

✍️ Author
Aman Tiwari
