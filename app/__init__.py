import os
from flask import Flask, render_template, request, flash, redirect, url_for
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
# from flask_wtf import RecaptchaField # Supprimé: Plus besoin de RecaptchaField
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Email, Length
import requests
import json # Gardé car potentiellement utilisé pour d'autres appels API si besoin

# Load environment variables from the .env file
load_dotenv()

# Create the Flask application instance here, and expose it directly
# instead of returning it from a create_app function.
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super_secret_key_fallback'),
    # Supprimé: Plus besoin des clés reCAPTCHA dans la configuration
    # RECAPTCHA_PUBLIC_KEY=os.getenv('RECAPTCHA_SITE_KEY'),
    # RECAPTCHA_PRIVATE_KEY=os.getenv('RECAPTCHA_SECRET_KEY')
)

# Supprimé: Les logs de débogage pour reCAPTCHA ne sont plus nécessaires
# print(f"DEBUG_FLASK: SECRET_KEY loaded (first 5 chars): {app.config['SECRET_KEY'][:5]}...")
# print(f"DEBUG_FLASK: RECAPTCHA_PUBLIC_KEY loaded: {app.config['RECAPTCHA_PUBLIC_KEY']}")
# print(f"DEBUG_FLASK: RECAPTCHA_PRIVATE_KEY loaded (last 5 chars): {app.config['RECAPTCHA_PRIVATE_KEY'][-5:] if app.config['RECAPTCHA_PRIVATE_KEY'] else 'None'}")


# Initialize CSRF protection
CSRFProtect(app)

# Import and register blueprints *after* the 'app' instance is created
# and its configurations and extensions are initialized
from . import routes # Imports the routes module
app.register_blueprint(routes.bp) # Registers the blueprint

# Définition du formulaire de contact mise à jour (sans RecaptchaField)
class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    subject = StringField('Subject', validators=[DataRequired(), Length(min=2, max=200)])
    message = TextAreaField('Message', validators=[DataRequired(), Length(min=10, max=1000)])
    # Supprimé: Plus de champ reCAPTCHA
    # recaptcha = RecaptchaField()
    submit = SubmitField('Send Message')


# Static data for the portfolio (inchangé)
portfolio_data = {
    "skills": {
        "Programming Languages": [
            {"name": "Python", "icon": "fab fa-python"},
            {"name": "SQL", "icon": "fas fa-database"},
            {"name": "JavaScript", "icon": "fab fa-js"},
            {"name": "HTML5", "icon": "fab fa-html5"},
            {"name": "CSS3", "icon": "fab fa-css3-alt"}
        ],
        "Machine Learning & AI": [
            {"name": "Scikit-learn", "icon": "fas fa-brain"},
            {"name": "TensorFlow", "icon": "fas fa-cube"},
            {"name": "PyTorch", "icon": "fas fa-fire"},
            {"name": "Keras", "icon": "fas fa-code"},
            {"name": "OpenCV", "icon": "fas fa-camera"},
            {"name": "NLTK", "icon": "fas fa-language"},
            {"name": "SpaCy", "icon": "fas fa-comment-alt"},
            {"name": "Generative AI", "icon": "fas fa-robot"},
            {"name": "LLMs (Gemini, LangChain, RAG)", "icon": "fas fa-comments"},
        ],
        "Data Science & Analysis": [
            {"name": "NumPy", "icon": "fas fa-calculator"},
            {"name": "Pandas", "icon": "fas fa-table"},
            {"name": "Matplotlib", "icon": "fas fa-chart-bar"},
            {"name": "Seaborn", "icon": "fas fa-chart-line"},
            {"name": "Dash/Plotly", "icon": "fas fa-chart-area"},
            {"name": "Jupyter", "icon": "fas fa-book"}
        ],
        "MLOps & Deployment": [
            {"name": "Docker", "icon": "fab fa-docker"},
            {"name": "Git/GitHub", "icon": "fab fa-github"},
            {"name": "CI/CD", "icon": "fas fa-sync-alt"},
            {"name": "FastAPI", "icon": "fas fa-network-wired"},
            {"name": "Streamlit", "icon": "fas fa-stream"},
            {"name": "Render", "icon": "fas fa-cloud"}
        ],
        "Databases & Cloud": [
            {"name": "PostgreSQL", "icon": "fas fa-database"},
            {"name": "MySQL", "icon": "fas fa-database"},
            {"name": "AWS (basics)", "icon": "fab fa-aws"}
        ]
    },
    "experience": [
        {
            "year": "Mars 2025 – Juin 2025",
            "title": "Machine Learning Engineer Intern",
            "company": "M2I – France (Remote)",
            "description": [
                "Developed an AI-Powered Algorithmic Trading Robot, integrating NewsAPI, Google Gemini for real-time sentiment analysis, and a FAISS-based RAG system to generate justified trading signals. (See portfolio for demo).",
                "Led a comprehensive predictive modeling project for an agricultural data management system, covering data collection, processing, visualization, and prediction of agricultural yield.",
                "Designed and optimized supervised Machine Learning models (Random Forest, XGBoost, Logistic Regression).",
                "Implemented a robust data pipeline using FastAPI, PostgreSQL, Streamlit and deployed the solution with Docker on Heroku (now Render), ensuring full version control via GitHub.",
                "Continuously optimized model performance (F1-score tuning, drift monitoring), enhancing prediction reliability."
            ]
        },
        {
            "year": "Septembre 2023 – Juin 2024",
            "title": "Machine Learning Engineer Intern",
            "company": "Divination – New York, NY (Remote)",
            "description": [
                "Developed a Personal CV RAG Chatbot and a Banking RAG Chatbot. These projects demonstrated my ability to extract contextual information from documents (PDF) and knowledge bases for precise responses using LangChain and Google Gemini. (Demos available on my portfolio).",
                "Built a sentiment analysis system for customer complaints, contributing to a 57% increase in the success of targeted marketing campaigns.",
                "Developed a market intelligence tool using NLP and Deep Learning to analyze customer behavior.",
                "Designed a customer behavior prediction algorithm based on LSTM with TensorFlow, increasing market share by 27%."
            ]
        },
        {
            "year": "Mars 2021 – Août 2023",
            "title": "Chef d'équipe / Opérations",
            "company": "Amazon - Brieselang, Allemagne",
            "description": [
                "Managed a team of 45-55 associates, overseeing logistics operations and inventory management for a 200,000 m² warehouse.",
                "Conducted daily data analyses to identify deviations and implement corrective actions, maintaining a strong grasp of KPIs and budget control.",
                "Acted as a crucial link between management and team members, ensuring operational efficiency and associate development."
            ]
        }
    ],
    "education": [
        {"year": "Sept. 2023 – Sept. 2025", "title": "Master of Science in Artificial Intelligence", "institution": "University of London (Remote)"},
        {"year": "Oct. 2024 – Août 2025", "title": "Data Scientist & ML Engineer Training", "institution": "M2I Formation, Paris (Remote)"},
        {"year": "Sept. 2020 – Oct. 2023", "title": "Bachelor of Science in Computer Science", "institution": "University of London (Remote)"},
        {"year": "Sept. 2017 – Sept. 2020", "title": "BTS Management des Unités Commerciales", "institution": "Lycée Sainte Marie, France"}
    ],
    "certifications": [
        "Data Scientist (M2I Formation)",
        "Machine Learning Engineer (M2I Formation)",
        "Deep Learning Specialization (Coursera - Andrew Ng)",
        "TensorFlow Developer (Google)",
        "Microsoft Certified: Azure AI Fundamentals",
        "Scrum Master Certified (Scrum.org)"
    ],
    "languages": [
        "French (Native)",
        "English (Fluent)",
        "German (Professional Working Proficiency)"
    ],
    "projects": [
        {
            "id": "llm_trading_robot",
            "title": "AI-Powered Algorithmic Trading Robot (RAG & Sentiment Analysis)",
            "category": "LLM / Generative AI",
            "image": "trading_robot_ai.png",
            "description_short": "Developed an AI trading robot that generates BUY/HOLD/SELL signals by analyzing real-time financial news sentiment via Google Gemini and a FAISS-based RAG system. Deployed on Hugging Face Spaces.",
            "description_long": "This project focuses on building an intelligent algorithmic trading robot that leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) for real-time market analysis. The robot integrates with a news API to fetch the latest financial news, processes it through Google Gemini for sentiment analysis, and then uses a FAISS vector database in a RAG pipeline to provide context-aware trading signals (BUY/HOLD/SELL). The goal is to reduce market risk and enhance decision-making through AI-driven insights. The application is deployed via Hugging Face Spaces, ensuring accessibility and demonstrating scalable deployment capabilities.",
            "technologies": ["Python", "LangChain", "Google Gemini", "FAISS", "NewsAPI", "Gradio"],
            "github_link": "https://github.com/Amen-a-setheru/AI_Trading_Robot",
            "demo_link": "https://huggingface.co/spaces/Amenastheru/AI_Trading_Robot"
        },
        {
            "id": "llm_personal_cv_chatbot",
            "title": "Personal CV RAG Chatbot",
            "category": "LLM / Generative AI",
            "image": "cv_chatbot.png",
            "description_short": "Designed and implemented an interactive chatbot capable of precisely answering questions about a PDF CV by utilizing a RAG architecture with LangChain and Google Gemini. Deployed on Hugging Face Spaces.",
            "description_long": "This project involved creating a personalized chatbot that acts as an intelligent assistant for my CV. It uses a Retrieval Augmented Generation (RAG) framework, specifically LangChain, to extract information from my CV (stored as a PDF) and respond to user queries accurately and contextually. The integration with Google Gemini ensures highly relevant and natural language responses. This demonstrates expertise in NLP, LLM fine-tuning, and building practical AI applications.",
            "technologies": ["Python", "LangChain", "Google Gemini", "FAISS", "PyPDFLoader", "Gradio"],
            "github_link": "https://github.com/Amen-a-setheru/CV-RAG-CHATBOT",
            "demo_link": "https://huggingface.co/spaces/Amenastheru/CV-RAG-CHATBOT"
        },
        {
            "id": "llm_banking_chatbot",
            "title": "Banking RAG Chatbot",
            "category": "LLM / Generative AI",
            "image": "banking_chatbot.png",
            "description_short": "Developed a Banking Chatbot using RAG and Google Gemini to provide accurate and secure answers to customer banking queries from a knowledge base. Deployed on Hugging Face Spaces.",
            "description_long": "This project developed a robust banking chatbot designed to assist customers with their queries by leveraging a Retrieval Augmented Generation (RAG) architecture. It connects to a predefined banking knowledge base to fetch relevant information and uses Google Gemini to formulate precise and helpful responses. The focus was on ensuring accuracy, security, and a seamless user experience, demonstrating the power of LLMs in customer service applications.",
            "technologies": ["Python", "LangChain", "Google Gemini", "FAISS", "Gradio"],
            "github_link": "https://github.com/Amen-a-setheru/BANKING-RAG-CHATBOT",
            "demo_link": "https://huggingface.co/spaces/Amenastheru/BANKING-RAG-CHATBOT"
        },
        {
            "id": "sales_forecasting",
            "title": "Sales Forecasting Model",
            "category": "Classical Machine Learning",
            "image": "sales_forecast.png",
            "description_short": "Built a machine learning model to predict sales, incorporating time-series data and external factors for improved accuracy.",
            "description_long": "Developed a robust sales forecasting model using advanced time-series analysis techniques and machine learning algorithms (e.g., ARIMA, Prophet, XGBoost). The model processes historical sales data, promotional calendars, and external economic indicators to provide highly accurate sales predictions. This project involved extensive data cleaning, feature engineering, model training, validation, and hyperparameter tuning to ensure optimal performance. The forecasts enable better inventory management and resource allocation.",
            "technologies": ["Python", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "Prophet", "Matplotlib"],
            "github_link": "https://github.com/Amen-a-setheru/Sales_Forecasting",
            "demo_link": None
        },
        {
            "id": "customer_churn",
            "title": "Customer Churn Prediction",
            "category": "Classical Machine Learning",
            "image": "churn_prediction.png",
            "description_short": "Implemented a customer churn prediction system, identifying at-risk customers to enable proactive retention strategies.",
            "description_long": "Designed and deployed a machine learning model to predict customer churn for a telecom company. The project involved data preprocessing, feature selection (using techniques like ANOVA and RFE), and training various classification models (Logistic Regression, Random Forest, Gradient Boosting). The model identifies customers likely to churn, allowing the business to implement targeted retention campaigns, significantly reducing customer attrition. Key metrics like precision, recall, and F1-score were optimized.",
            "technologies": ["Python", "Pandas", "Scikit-learn", "Seaborn", "Matplotlib", "Jupyter"],
            "github_link": "https://github.com/Amen-a-setheru/Customer-Churn-Prediction",
            "demo_link": None
        },
        {
            "id": "image_classification",
            "title": "Image Classification with Deep Learning",
            "category": "Deep Learning / Computer Vision",
            "image": "image_classification.png",
            "description_short": "Developed a deep learning model for image classification, achieving high accuracy on a custom dataset of various objects.",
            "description_long": "This project involved building and training a Convolutional Neural Network (CNN) using TensorFlow and Keras for image classification. The goal was to accurately categorize images from a diverse custom dataset (e.g., different types of animals, objects, or scenes). I focused on data augmentation, transfer learning (using pre-trained models like VGG16 or ResNet), and fine-tuning to achieve robust performance and high accuracy. The project demonstrates strong skills in computer vision and deep learning model development.",
            "technologies": ["Python", "TensorFlow", "Keras", "OpenCV", "Matplotlib"],
            "github_link": "https://github.com/Amen-a-setheru/image_classification",
            "demo_link": None
        }
    ]
}

@app.route('/')
def index():
    # Passe app.config à la template pour les cas où vous avez des variables de configuration
    # que vous voulez utiliser dans le frontend (comme la clé publique reCAPTCHA si vous la réactivez un jour)
    return render_template('index.html', **portfolio_data, form=ContactForm(), config=app.config)

@app.route('/project/<id>')
def project_detail(id):
    project = next((p for p in portfolio_data["projects"] if p["id"] == id), None)
    if project:
        return render_template('project_detail.html', project=project)
    return "Project not found", 404

@app.route('/contact', methods=['GET', 'POST'])
def contact_submit():
    form = ContactForm() # Utilisez l'instance du formulaire mise à jour (sans reCAPTCHA)
    if form.validate_on_submit():
        # Supprimé: Plus de vérification reCAPTCHA
        # recaptcha_response = request.form.get('g-recaptcha-response')
        # secret_key = app.config['RECAPTCHA_PRIVATE_KEY']
        # ... (tout le bloc de vérification reCAPTCHA)

        flash('Your message has been sent successfully!', 'success')
        # Ici, vous enverriez normalement l'e-mail
        # Par exemple: send_email(form.name.data, form.email.data, form.subject.data, form.message.data)
        return redirect(url_for('index') + '#contact-form-section') # Redirection vers la section contact
    
    # Si la requête est GET ou la validation a échoué (sans reCAPTCHA)
    return render_template('index.html', **portfolio_data, form=form, config=app.config)

# Note: Si vous avez un fichier `routes.py` et que ces routes y sont définies,
# assurez-vous de supprimer ces définitions ici et de les ajuster là-bas.
# Selon votre `__init__.py` fourni, `contact_submit` est bien ici.

# Si vous avez un bloc 'if __name__ == "__main__":' en bas de ce fichier,
# assurez-vous qu'il est supprimé ou commenté pour Render.
