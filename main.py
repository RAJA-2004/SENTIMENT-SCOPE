import tkinter as tk  # For GUI
from tkinter import font as tkfont
from tkinter import messagebox
import pickle  # For saving & loading complex datatype
import re  # Pattern matching in strings
from nltk.corpus import stopwords  # Removes common stopwords
from nltk.stem.porter import PorterStemmer  # Reduces words to root form

# Load trained ML model and vectorizer
with open('model/trained_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing
port_stem = PorterStemmer()


def stemming(content):
    """
    Preprocess the input text by removing non-alphabetic characters,
    converting to lowercase, removing stop words, and applying stemming.
    """
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def predict_sentiment(sentence):
    """
    Predict the sentiment of the given sentence using the pre-trained model and TF-IDF vectorizer.
    """
    preprocessed_sentence = stemming(sentence)
    transformed_sentence = vectorizer.transform([preprocessed_sentence])
    prediction = model.predict(transformed_sentence)
    return 'Positive' if prediction[0] == 1 else 'Negative'


def check_contradictory_phrases(sentence):
    """
    Check if the sentence contains contradictory phrases.
    """
    contradictory_phrases = [
        "not good", "not great", "not bad", "not terrible", "not excellent", "not amazing"
    ]
    for phrase in contradictory_phrases:
        if phrase in sentence.lower():
            return True
    return False


def on_predict_button_click():
    """
    Handle the click event for the Predict Sentiment button.
    Retrieve the input text, perform sentiment prediction, and update the result label.
    """
    sentence = text_entry.get("1.0", tk.END).strip()

    if not sentence:
        messagebox.showwarning("Input Error", "Please enter a sentence to analyze.")
        return

    if check_contradictory_phrases(sentence):
        messagebox.showinfo("Contradictory Sentiment", "The sentence contains both positive and negative sentiments.")
        result_label.config(text="Sentiment: Mixed")
        return

    sentiment = predict_sentiment(sentence)
    result_label.config(text=f"Sentiment: {sentiment}")


# Main Tkinter window

root = tk.Tk()
root.title("Sentiment Analysis")
window_size = 350
root.geometry(f"{window_size}x{window_size}")
default_font = tkfont.Font(size=15)
sentiment_font = tkfont.Font(size=20)

tk.Label(root, text="Enter your sentence:", font=default_font).pack(pady=10)
text_entry = tk.Text(root, height=5, width=25, font=default_font)
text_entry.pack(pady=10)

# Button to trigger sentiment prediction
predict_button = tk.Button(root, text="Predict Sentiment", command=on_predict_button_click, font=default_font, height=2, width=20)
predict_button.pack(pady=10)

# Label to display the predicted sentiment
result_label = tk.Label(root, text="Sentiment: ", font=sentiment_font)
result_label.pack(pady=10)

root.mainloop()
