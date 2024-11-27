# ML model to classify urls bases on a generated csv file with a list of legit and phishing urls


# combined_urls.csv (DATASET)

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


# MODEL AND UI

    import pandas as pd
    import re
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    import tkinter as tk
    from tkinter import messagebox, ttk
    
    # Load the dataset
    df = pd.read_csv("combined_urls.csv")
    X = df['URL']
    y = df['Label']
    
    # Feature extraction function
    def url_features(url):
        return [
            len(url),
            url.count("."),
            url.count("-"),
            url.count("@"),
            url.count("//"),
            1 if "https" in url else 0,
            1 if re.search(r"(bit\.ly|t\.co|tinyurl)", url) else 0,
            sum(1 for char in url if char.isdigit()),
            any(keyword in url.lower() for keyword in ["login", "verify", "account", "update", "bank"])
        ]
    
    class URLFeaturesExtractor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            return [url_features(url) for url in X]
    
    pipeline = Pipeline([
        ('features', URLFeaturesExtractor()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, "phishing_model.pkl")
    model = joblib.load("phishing_model.pkl")
    
    def predict_url(url):
        prediction = model.predict([url])[0]
        return "Phishing" if prediction == 1 else "Legitimate"
    
    def check_url():
        url = url_entry.get()
        if url:
            result = predict_url(url)
            messagebox.showinfo("Prediction Result", f"The URL is: {result}")
        else:
            messagebox.showwarning("Input Error", "Please enter a URL.")
    
    # Create the UI
    root = tk.Tk()
    root.title("Phishing URL Detector")
    root.geometry("400x300")
    root.configure(bg="#f0f0f0")
    
    # Create a style
    style = ttk.Style()
    style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
    style.configure("TButton", padding=6, relief="flat", background="#000000", foreground="#0000FF", font=("Helvetica", 12))  # Change button color here
    style.map("TButton", background=[("active", "#0056b3")])  # Change active color here
    
    # Create UI elements
    ttk.Label(root, text="Enter URL:", font=("Helvetica", 14)).pack(pady=20)
    url_entry = ttk.Entry(root, width=50)
    url_entry.pack(pady=5)
    
    check_button = ttk.Button(root, text="Check URL", command=check_url)
    check_button.pack(pady=20)
    
    # Run the application
    root.mainloop()


# CLI TOOL:

# phishing_checker.py


    import requests
    import pandas as pd
    import re
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
    import argparse
    
    class URLFeaturesExtractor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
    
        def transform(self, X, y=None):
            return [self.url_features(url) for url in X]
    
        @staticmethod
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
    
    def load_model(model_path):
        return joblib.load(model_path)
    
    def predict_url(model, url):
        prediction = model.predict([url])[0]
        return "Phishing" if prediction == 1 else "Legitimate"
    
    def main(url):
        model = load_model("phishing_model.pkl")
        result = predict_url(model, url)
        print(f"The URL '{url}' is: {result}")
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Check if a URL is phishing or legitimate.")
        parser.add_argument("url", type=str, help="The URL to check")
        args = parser.parse_args()
        
        main(args.url)

# setup.py

    from setuptools import setup
    
    setup(
        name='phishing_checker',
        version='0.1',
        py_modules=['phishing_checker'],
        install_requires=[
            'requests',
            'pandas',
            'scikit-learn',
            'joblib',
        ],
        entry_points={
            'console_scripts': [
                'phishing-checker=phishing_checker:main',
            ],
        },
    )

# INSTALLING AND USING THE TOOL:

    pip install .

    python phishing_checker.py "[URL]"


# OUTPUT

![image](https://github.com/user-attachments/assets/61edfe1a-5a7e-4cf7-845d-48d6d8d18d5d)

![image](https://github.com/user-attachments/assets/0b1dc2ee-8aea-4dc8-8335-fb56a1c02b04)


![Screenshot 2024-11-27 181851](https://github.com/user-attachments/assets/30a71229-9c77-4bf2-983c-0a7c73b2ee2c)

![Screenshot 2024-11-27 210516](https://github.com/user-attachments/assets/43d6182f-57ab-41bd-bb70-d116b75d0592)









