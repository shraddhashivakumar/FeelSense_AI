FeelSense AI: Emotion-driven Conversational AI

This project is a Flask-based web application that serves a simple chatbot. It uses a custom Scikit-learn model (SGD Classifier) to perform real-time sentiment analysis on user input and switches its conversational mode (Therapy or Education) accordingly.

The model has been optimized with balanced class weights to address class imbalance and accurately identify challenging negative phrases, such as "exam tension."

Getting Started:

Follow these steps to clone the repository, set up your environment, and run the application.

Prerequisites:

You need Python 3.8+ installed on your system.

1. Clone the Repository

Open your terminal or command prompt and clone the project

2. Set Up the Environment

It is highly recommended to use a virtual environment to manage dependencies:

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt


3. Run the Flask Server

Once dependencies are installed, you can start the application server:

python app.py


The application will now be running. You can access it in your web browser at: http://127.0.0.1:5000

Retraining the Mood Model

The heart of this application is its custom sentiment model. If you want to improve accuracy or introduce new mood labels, follow this process:

1. Update the Dataset

Modify the file located at data/emotion.csv.

Format: Each line must be: sentence,mood_label.

Goal: Add more examples, especially for moods that the model struggles with (like fear or disappointment).

2. Run the Training Script

The train_and_deploy.py script automatically uses the updated data, applies the optimized SGD model, and saves the new, updated model files (.joblib).

python train_and_deploy.py


Output: The script will print a Classification Report showing the model's performance metrics before saving the new files.

3. Deploy the New Model

After training, you must restart the Flask server so it loads the new model files into memory:

Stop the server (Press CTRL+C in the server terminal).

Start the server again: python app.py

Your chatbot will now be using the improved, retrained model!
