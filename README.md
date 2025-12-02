# 📧 Email/SMS Spam Classifier

A machine learning-powered web application that classifies emails and SMS messages as spam or legitimate (ham) using Natural Language Processing techniques and the Multinomial Naive Bayes algorithm.

## 🚀 Live Demo

**[Try it here!](https://email-spam-classifier-ml-model.onrender.com)**

## 📋 Description

This project implements an intelligent spam detection system that analyzes text messages and emails to determine whether they are spam or not. The application uses advanced text preprocessing techniques including tokenization, stemming, and stopword removal, combined with TF-IDF vectorization and the MultinomialNB classifier to achieve high accuracy in spam detection.

## ✨ Features

- 🎯 Real-time spam detection for emails and SMS messages
- 🧹 Advanced text preprocessing (lowercasing, tokenization, stemming)
- 🔤 TF-IDF vectorization for feature extraction
- 🤖 Multinomial Naive Bayes classification algorithm
- 🌐 Interactive web interface built with Streamlit
- ⚡ Fast and accurate predictions
- 📱 Responsive design for all devices

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **NLTK** - Natural language processing
- **Scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation
- **Pickle** - Model serialization

## 📊 Model Performance

The MultinomialNB classifier was chosen after comparing multiple algorithms including:

- Gaussian Naive Bayes
- Multinomial Naive Bayes (selected for best performance)
- Bernoulli Naive Bayes

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/RifatHossaiN47/Email-Spam-Classifier-ML-Model.git
   cd Email-Spam-Classifier-ML-Model
   ```

2. **Create a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## 🚀 Usage

1. **Run the Streamlit application**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter a message** in the text input field

4. **Click "Predict"** to classify the message as SPAM or NOT SPAM

## 📁 Project Structure

```
Email-Spam-Classifier/
│
├── app.py                          # Main Streamlit application
├── SpamMessageClassifier.ipynb     # Jupyter notebook with model training
├── model1.pkl                      # Trained MultinomialNB model
├── vectorizer.pkl                  # TF-IDF vectorizer
├── requirements.txt                # Python dependencies
├── nltk.txt                        # NLTK data requirements
└── README.md                       # Project documentation
```

## 🔍 How It Works

1. **Text Preprocessing**

   - Convert text to lowercase
   - Tokenize the text into words
   - Remove punctuation and non-alphanumeric characters
   - Remove stopwords (common words like "the", "is", etc.)
   - Apply Porter Stemming to reduce words to their root form

2. **Feature Extraction**

   - Use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
   - Convert preprocessed text into numerical features

3. **Classification**
   - Feed the vectorized text to the trained MultinomialNB model
   - Get prediction: SPAM (1) or NOT SPAM (0)

## 📦 Dependencies

Key libraries used in this project:

- `streamlit==1.38.0`
- `nltk==3.9.1`
- `scikit-learn==1.3.2`
- `pandas==2.0.3`
- `numpy==1.24.4`

For a complete list, see `requirements.txt`

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Rifat Hossain**

- GitHub: [@RifatHossaiN47](https://github.com/RifatHossaiN47)

## 🙏 Acknowledgments

- Dataset used for training the model
- NLTK for natural language processing tools
- Streamlit for the amazing web framework
- Scikit-learn for machine learning algorithms

---

⭐ If you found this project helpful, please consider giving it a star!
