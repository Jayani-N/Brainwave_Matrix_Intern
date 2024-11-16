# ML model to classify urls bases on a generated csv file with a list of legit and phishing urls


# phishing_urls.csv


    import requests
    import pandas as pd
    
    # Fetch URLs from OpenPhish (direct link)
    phishing_urls = []
    try:
        response = requests.get("https://openphish.com/feed.txt")
        if response.status_code == 200:
            phishing_urls = response.text.splitlines()
        else:
            print("Failed to retrieve OpenPhish data.")
    except Exception as e:
        print(f"Error fetching OpenPhish data: {e}")
    
    # Label all fetched URLs as phishing (Label: 1)
    phishing_data = {"URL": phishing_urls, "Label": [1] * len(phishing_urls)}
    df_phishing = pd.DataFrame(phishing_data)
    print(f"Phishing URLs collected: {len(df_phishing)}")
    
    # Save the DataFrame to a CSV file
    df_phishing.to_csv("phishing_urls.csv", index=False)
    print("Phishing URLs saved to phishing_urls.csv")

# legitimate_urls.csv

    import requests
    import pandas as pd
    
    # Cisco Umbrella top domains URL
    url = "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
    
    try:
        # Download the zipped CSV file
        response = requests.get(url)
        if response.status_code == 200:
            with open("top-1m.csv.zip", "wb") as file:
                file.write(response.content)
    
            # Unzip and read the file into a DataFrame
            df_legit = pd.read_csv("top-1m.csv.zip", compression="zip", header=None, names=["Rank", "URL"])
    
            # Keep only the URL column and add the Label column
            df_legit = df_legit[["URL"]]
            df_legit["Label"] = 0
    
            # Save to a CSV file
            df_legit.to_csv("legitimate_urls.csv", index=False)
            print("Legitimate URLs saved to legitimate_urls.csv")
        else:
            print("Failed to retrieve Cisco Umbrella data.")
    except Exception as e:
        print(f"Error fetching Cisco Umbrella data: {e}")

# combined_urls.csv

    import requests
    import pandas as pd
    
    # Fetch URLs from OpenPhish (direct link) and label them as phishing (1)
    phishing_urls = []
    try:
        response = requests.get("https://openphish.com/feed.txt")
        if response.status_code == 200:
            phishing_urls = response.text.splitlines()
        else:
            print("Failed to retrieve OpenPhish data.")
    except Exception as e:
        print(f"Error fetching OpenPhish data: {e}")
    
    # Label phishing URLs as 1
    phishing_data = {"URL": phishing_urls, "Label": [1] * len(phishing_urls)}
    df_phishing = pd.DataFrame(phishing_data)
    print(f"Phishing URLs collected: {len(df_phishing)}")
    
    # Fetch Cisco Umbrella top domains (legitimate URLs) and label them as legitimate (0)
    url = "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open("top-1m.csv.zip", "wb") as file:
                file.write(response.content)
    
            # Unzip and read the file into a DataFrame
            df_legit = pd.read_csv("top-1m.csv.zip", compression="zip", header=None, names=["Rank", "URL"])
    
            # Keep only the URL column and add the Label column (0 for legitimate)
            df_legit = df_legit[["URL"]]
            df_legit["Label"] = 0
            print(f"Legitimate URLs collected: {len(df_legit)}")
        else:
            print("Failed to retrieve Cisco Umbrella data.")
    except Exception as e:
        print(f"Error fetching Cisco Umbrella data: {e}")
    
    # Combine both phishing and legitimate URLs
    df_combined = pd.concat([df_phishing, df_legit], ignore_index=True)
    
    # Save the combined dataset to a CSV file
    df_combined.to_csv("combined_urls.csv", index=False)
    print("Combined dataset saved as combined_urls.csv")


# MODEL

    import pandas as pd
    import re
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    
    # Load the dataset (ensure your CSV file has 'URL' and 'Label' columns)
    df = pd.read_csv("combined_urls.csv")  # Replace with your dataset file
    X = df['URL']
    y = df['Label']
    
    # Feature extraction function
    def url_features(url):
        return [
            len(url),                                     # URL length
            url.count("."),                               # Number of dots
            url.count("-"),                               # Number of hyphens
            url.count("@"),                               # Presence of @ symbol
            url.count("//"),                              # Number of slashes
            1 if "https" in url else 0,                   # Check for HTTPS
            1 if re.search(r"(bit\.ly|t\.co|tinyurl)", url) else 0,  # Shortened URL
            sum(1 for char in url if char.isdigit()),     # Number of digits
            any(keyword in url.lower() for keyword in ["login", "verify", "account", "update", "bank"])  # Suspicious keywords
        ]
    
    # Custom transformer to apply feature extraction
    class URLFeaturesExtractor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            return [url_features(url) for url in X]
    
    # Define a pipeline with feature extraction and Random Forest model
    pipeline = Pipeline([
        ('features', URLFeaturesExtractor()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = pipeline.predict(X_test)
    print("Model accuracy:", accuracy_score(y_test, y_pred))
    
    # Save the model
    joblib.dump(pipeline, "phishing_model.pkl")
    
    import joblib
    
    # Load the trained model
    model = joblib.load("phishing_model.pkl")
    
    # Predict function for a new URL
    def predict_url(url):
        prediction = model.predict([url])[0]
        result = "Phishing" if prediction == 1 else "Legitimate"
        return result
    
    # Example prediction
    url = input("Enter the url:")
    print(f"The URL is: {predict_url(url)}")

# OUTPUT

![URL_SCANNER_1](https://github.com/user-attachments/assets/13ae2dfe-54dd-4ee0-af47-409c09ce1e1a)

![OUTPUT1](https://github.com/user-attachments/assets/62ec5ea1-b26d-4b30-83a2-c112449c9257)

![URL_SCANNER2](https://github.com/user-attachments/assets/884009bb-def3-4c6d-b5bb-bb303393bcce)

![OUTPUT2](https://github.com/user-attachments/assets/9f0ff652-aaa8-43a9-9368-053c34fc10f6)




