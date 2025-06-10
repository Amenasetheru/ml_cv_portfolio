import os
import requests
from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, current_app
from .forms import ContactForm


bp = Blueprint('main', __name__)

# --- Data for the Portfolio ---

# Project Data (English version, based on your CV)
PROJECTS = [
    {
        'id': 'sentiment-analysis',
        'title': 'Sentiment Analysis with DistilBERT and PyTorch',
        'category': 'NLP',
        'image': 'senti_analy10.jpeg', # Filename only
        'description_short': "Transformers model for customer review sentiment analysis, achieving 95% F1-score.",
        'description_long': """
            Developed an advanced Deep Learning model using DistilBERT for sentiment analysis from online customer reviews.
            The process included meticulous text data preprocessing (tokenization, cleaning), model training with PyTorch,
            and achieving 95% accuracy and F1-score on the test set.
            The user interface was deployed via Hugging Face Spaces for easy demonstration.
        """,
        'technologies': ['Python', 'PyTorch', 'Hugging Face', 'DistilBERT', 'NLP'],
        'github_link': 'https://github.com/Amenasetheru/Deep-Learning-Projects/blob/main/Sentiment_Analysis_with_Pytorch_using_DistilBERT_from_Hugging_Face_.ipynb',
        'demo_link': 'https://huggingface.co/spaces/votre_profil/sentiment_analysis_demo' # Update this link with your actual HF Spaces link
    },
    {
        'id': 'marketing-optimization',
        'title': 'Recommender System with TensorFlow',
        'category': 'Marketing ML',
        'image': 'recom_syst5.jpeg', # Filename only
        'description_short': "Recommendation and sentiment analysis models for enhanced marketing efficiency.",
        'description_long': """
            Developed innovative models for product recommendation, customer retention, and sentiment analysis,
            leveraging the power of NLP and Deep Learning. These solutions increased marketing campaign efficiency by 57%
            and customer loyalty by 12%. Models were deployed using TensorFlow and PyTorch frameworks.
        """,
        'technologies': ['Python', 'TensorFlow', 'PyTorch', 'NLP', 'Deep Learning'],
        'github_link': 'https://github.com/Amenasetheru/Deep-Learning-Projects/blob/main/Recommender_system_with_Tensorflow_using_Amazon_Product_Reviews.ipynb',
        'demo_link': None
    },
    {
        'id': 'fraud-detection',
        'title': 'Credit Card Fraud Detection with CNN',
        'category': 'Computer Vision', # This project is usually Tabular Data/Anomaly Detection, but I'll stick to your classification
        'image': 'credi_card_detec4.jpeg', # Filename only
        'description_short': "CNN model for fraud detection on imbalanced transactional datasets.",
        'description_long': """
            Developed a Convolutional Neural Network (CNN) model for fraud detection on a large transactional dataset.
            A strong emphasis was placed on data cleaning and preprocessing, as well as handling highly imbalanced data
            through advanced resampling techniques. The model achieved 95% accuracy in fraud detection.
        """,
        'technologies': ['Python', 'TensorFlow', 'CNN', 'Data Preprocessing', 'Imbalanced Data'],
        'github_link': 'https://github.com/Amenasetheru/Deep-Learning-Projects/blob/main/Project_4_Credit_Card_Fraud_Detection_with_CNN_AAH.ipynb',
        'demo_link': None
    },
    {
        'id': 'bank-satisfaction',
        'title': 'Bank Customer Satisfaction Prediction',
        'category': 'Machine Learning',
        'image': 'cust_satis_pred7.jpeg', # Filename only
        'description_short': "CNN model predicting bank customer satisfaction from transactional data.",
        'description_long': """
            Cleaned and preprocessed a comprehensive dataset of bank customer transactions.
            Performed in-depth exploratory data analysis to uncover key insights.
            Built a Convolutional Neural Network (CNN) model designed to predict customer satisfaction,
            demonstrating the ability to identify factors influencing customer sentiment.
        """,
        'technologies': ['Python', 'TensorFlow', 'CNN', 'Data Preprocessing', 'EDA'],
        'github_link': 'https://github.com/Amenasetheru/Deep-Learning-Projects/blob/main/Predicting_the_Bank_Customer_Satisfaction_with_CNN.ipynb', # Add actual link
        'demo_link': None
    }
]

# Professional Experience Timeline Data
TIMELINE_EXPERIENCE = [
    {
        'year': 'Mar 2025 – Jun 2025',
        'title': 'Machine Learning Engineer Intern',
        'company': 'M2I – France (Remote)',
        'description': [
            "Led a predictive modeling project for an agricultural data management system (for a cooperative in Douala).",
            "Designed a complete system for data collection, processing, visualization, and yield prediction.",
            "Developed supervised learning models (Random Forest, XGBoost, Logistic Regression).",
            "Implemented a full data pipeline using FastAPI, PostgreSQL, and Streamlit.",
            "Deployed the solution using Docker & Heroku; maintained full version control via GitHub.",
            "Optimized model performance (F1-score tuning, retraining thresholds, drift monitoring)."
        ]
    },
    {
        'year': 'Sep 2023 – Jun 2024',
        'title': 'Machine Learning Engineer Intern',
        'company': 'Divination New York, NY (Remote)',
        'description': [
            "Built a sentiment analysis system to process customer complaints and identify product strengths, contributing to a 57% success rate in targeted marketing campaigns.",
            "Developed a market intelligence tool leveraging NLP and Deep Learning to understand customer behavior.",
            "Created a customer behavior prediction algorithm based on LSTM with TensorFlow, leading to a 27% increase in market share.",
            "Designed deep learning solutions that enhanced client understanding and improved customer retention by up to 12%."
        ]
    },
    {
        'year': 'Nov 2021 – Aug 2023',
        'title': 'Senior Key Account Manager',
        'company': 'Amazon.com Service LLC — New Jersey, USA',
        'description': [
            "Led strategic partnerships with 21+ companies (startups & brands).",
            "Designed growth initiatives and negotiated 40+ high-value contracts.",
            "Managed 7+ key accounts, aligning business goals with data-driven strategy.",
            "Experience in identifying client pain points—skills now redirected to ML problem-solving.",
            "Collaborated cross-functionally with technical and business teams (parallel to ML project work)."
        ]
    }
]

# Education Timeline Data
TIMELINE_EDUCATION = [
    {
        'year': '2024 – 2025',
        'title': 'Master’s Degree in Data Science & AI Engineering',
        'institution': 'M2i Formation, France',
    },
    {
        'year': '2010 – 2011',
        'title': 'Master’s in Business Development and Major Accounts',
        'institution': 'Neoma Business School, France',
    },
    {
        'year': '2010 – 2011',
        'title': 'Master’s in Political Sociology',
        'institution': 'Sorbonne University, Paris IV',
    },
    {
        'year': '2006 – 2008',
        'title': 'Bachelor of Arts in Political Science',
        'institution': 'New York University, NY, USA',
    },
    {
        'year': '2004 – 2004',
        'title': 'Certificate of English',
        'institution': 'Columbia University, NY, USA',
    }
]

# Skills Data - Removed 'level' property
SKILLS_DATA = {
    'Languages': [
        {'name': 'Python', 'icon': 'fab fa-python'},
        {'name': 'SQL', 'icon': 'fas fa-database'},
        {'name': 'Bash', 'icon': 'fas fa-terminal'},
        {'name': 'JavaScript (basics)', 'icon': 'fab fa-js'},
        {'name': 'Scala (basics)', 'icon': 'fas fa-code'},
        {'name': 'C (basics)', 'icon': 'fas fa-cogs'},
    ],
    'Libraries & Frameworks': [
        {'name': 'Scikit-learn', 'icon': 'fas fa-brain'},
        {'name': 'TensorFlow', 'icon': 'fas fa-cubes'},
        {'name': 'PyTorch', 'icon': 'fas fa-fire'},
        {'name': 'Hugging Face', 'icon': 'fas fa-comments'},
        {'name': 'XGBoost', 'icon': 'fas fa-chart-line'},
        {'name': 'Pandas & Numpy', 'icon': 'fas fa-table'},
        {'name': 'Matplotlib & Seaborn', 'icon': 'fas fa-chart-bar'},
    ],
    'NLP / Deep Learning': [
        {'name': 'Transformers (BERT, DistilBERT)', 'icon': 'fas fa-robot'},
        {'name': 'LSTM', 'icon': 'fas fa-stream'},
        {'name': 'CNN', 'icon': 'fas fa-camera'},
        {'name': 'TextVectorization, Embeddings', 'icon': 'fas fa-font'},
    ],
    'Data Engineering': [
        {'name': 'FastAPI', 'icon': 'fas fa-server'},
        {'name': 'Streamlit', 'icon': 'fas fa-desktop'},
        {'name': 'Docker', 'icon': 'fab fa-docker'},
        {'name': 'PostgreSQL', 'icon': 'fas fa-database'},
        {'name': 'Airflow', 'icon': 'fas fa-cloud-upload-alt'},
        {'name': 'Git', 'icon': 'fab fa-git-alt'},
    ],
    'Tools & Cloud': [
        {'name': 'VSCode', 'icon': 'fas fa-code'},
        {'name': 'GitHub', 'icon': 'fab fa-github'},
        {'name': 'Colab', 'icon': 'fas fa-laptop-code'},
        {'name': 'Heroku', 'icon': 'fas fa-cloud'},
        {'name': 'Hugging Face Spaces', 'icon': 'fas fa-space-shuttle'},
        {'name': 'AWS & Azure (basics)', 'icon': 'fab fa-aws'},
    ],
    'Methods': [
        {'name': 'Model Evaluation Classification (F1, Recall, ROC,)', 'icon': 'fas fa-chart-pie'},
        {'name': 'Model Evaluation Regression (R², RMSE, MAE)', 'icon': 'fas fa-chart-pie'},
        {'name': 'ML Ops', 'icon': 'fas fa-cogs'},
        {'name': 'CI/CD', 'icon': 'fas fa-sync-alt'},
    ],
    'Soft Skills': [
        {'name': 'Strategic thinking', 'icon': 'fas fa-lightbulb'},
        {'name': 'Client relations', 'icon': 'fas fa-handshake'},
        {'name': 'Communication', 'icon': 'fas fa-comments'},
        {'name': 'Leadership', 'icon': 'fas fa-user-tie'},
    ]
}

# Certification Data
CERTIFICATIONS = [
    "Practical Data Science – DeepLearning.AI Coursera (2022)",
    "AutoML & ML Pipelines – DeepLearning.AI Coursera (2022)",
    "Build, Train & Deploy ML with BERT – DeepLearning.AI Coursera (2022)",
    "TensorFlow: Advanced Techniques – DeepLearning.AI Coursera (2021)",
    "DeepLearning.AI TensorFlow Developer – Coursera (2021)",
    "NLP with TensorFlow – DeepLearning.AI Coursera (2021)",
    "Complete Data Science Bootcamp – Udemy (2020)"
]

# Language Data
LANGUAGES = [
    "English: Fluent – Lived and studied in the USA",
    "French: Native"
]


# --- Routes Definitions ---

@bp.route('/')
def index():
    form = ContactForm()
    return render_template('index.html',
                           projects=PROJECTS,
                           experience=TIMELINE_EXPERIENCE,
                           education=TIMELINE_EDUCATION,
                           skills=SKILLS_DATA,
                           certifications=CERTIFICATIONS,
                           languages=LANGUAGES,
                           form=form)

# Removed the /projects route as it's now integrated into index.html

@bp.route('/project/<project_id>')
def project_detail(project_id):
    project = next((p for p in PROJECTS if p['id'] == project_id), None)
    if project is None:
        flash("Project not found.", "error")
        return redirect(url_for('main.index', _anchor='projects-section')) # Redirect to projects section on home
    return render_template('project_detail.html', project=project)


@bp.route('/contact_submit', methods=['POST'])
def contact_submit():
    form = ContactForm()
    if form.validate_on_submit():
        recaptcha_response = request.form.get('g-recaptcha-response')

        if not recaptcha_response:
            flash('Please check the reCAPTCHA box.', 'error')
            return redirect(url_for('main.index', _anchor='contact-form-section'))

        secret_key = os.getenv('RECAPTCHA_PRIVATE_KEY')
        if not secret_key:
            current_app.logger.error("RECAPTCHA_PRIVATE_KEY not configured in environment variables.")
            flash('Server configuration error. Please try again later.', 'error')
            return redirect(url_for('main.index', _anchor='contact-form-section'))

        payload = {'secret': secret_key, 'response': recaptcha_response}
        r = requests.post('https://www.google.com/recaptcha/api/siteverify', data=payload)
        result = r.json()

        if result['success']:
            print(f"Message from {form.name.data} ({form.email.data}): {form.message.data}")
            flash('Your message has been sent successfully!', 'success')
            return redirect(url_for('main.index', _anchor='contact-form-section'))
        else:
            flash('reCAPTCHA verification failed. Please try again.', 'error')
            current_app.logger.warning(f"reCAPTCHA failed: {result.get('error-codes')}")
            return redirect(url_for('main.index', _anchor='contact-form-section'))
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"Error in {field.replace('_', ' ').title()}: {error}", 'error')
        return redirect(url_for('main.index', _anchor='contact-form-section'))


