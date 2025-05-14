# Project Portfolio

### 👋 Hello, I’m Mei, nice to meet you !

🙋‍♀️ I'm a data scientist with a background in __biology__ and a passion for solving complex problems with data. My path started in academic research, where I developed a human gene interaction database using BioPython to help researchers access critical information. This experience led me to __frontend development__ at Jimdo's Growth Team, where I built __data-driven A/B testing tools__ that increased conversion rates by 25% and discovered my love for extracting actionable insights from data. Now as a __full-fledged data scientist__, I uncover patterns, develop predictive models, and transform raw information into impactful solutions that drive decision-making.

__🧪 How I Think__
I thrive where data meets decisions. I approach problems methodically - from framing the right questions and preparing clean datasets to implementing appropriate models and measuring real-world impact. Whether it's engineering predictive features for credit risk assessment or building NLP pipelines for text classification, I focus on creating solutions that deliver tangible value and drive evidence-based decision making.

__🔎 Current Focus__
I’m diving deeper into applied __machine learning__, __deep learning__ and __model evaluation__. I’ve been building multi-label classifiers, experimenting with imbalance techniques, and fine-tuning large language models. I love pushing my limits and making things that are both smart and useful.

__🌱 Still Growing__
I'm constantly learning—whether it's exploring __AI agent__ architectures, fine-tuning large language models with __LoRA__, or learning __MLOps__ for seamless deployment. I'm also working on my __Dutch__ (currently approaching A2 fluency) while living in Belgium. I believe in learning in public and sharing knowledge, which is why I enjoy mentoring data science learners and leading interactive classroom sessions in my learning program.

__🤝 Let’s Connect__
If you’re into turning raw data into business insights, building predictive model for forecasting, or interesting AI product ideas, let's connect and have a chat !


## 🏷️ Skills Overview

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-2C3E50?style=for-the-badge&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)

## 📚 Categories

[Data Science](https://github.com/MeiChieh#-data-science)
- [Finance](https://github.com/MeiChieh#finance)
- [Health](https://github.com/MeiChieh#health)
- [Hobby](https://github.com/MeiChieh#hobby)
  
[Deep Learning](https://github.com/MeiChieh#-deep-learning)
- [NLP](https://github.com/MeiChieh#natural-language-processing)
- [Computer Vision](https://github.com/MeiChieh#computer-vision)

[Generative AI](https://github.com/MeiChieh#-generative-ai)
- [RAG chatbot](https://github.com/MeiChieh#-generative-ai)

### 🔬 Data Science

Projects involving data analysis, statistical modeling, and business insights.

#### Finance

- [**Home Credit Group Loan Default Risk Analysis and Modeling**](https://github.com/MeiChieh/home-credit-group-loan-default-prediction)  
  `Python` `NumPy` `LGBM` `Optuna` `Imblearn` `Error Analysis` `Shap`
  
  - Multi-table data aggregation and time series feature engineering
  - Combine boosting and logit model with soft voting to get robust result
  - Financial analysis to priopritize metrics and select the most optimal model

- [**Lending Club Loan Grade Analysis and Modeling**](https://github.com/MeiChieh/lending-club-loan-grade-prediction)  
  `Python` `Pandas` `Scikit-learn` `XGBoost` `Optuna` `Error Analysis` `Shap`
  
  - Financial data exploration and analysis
  - Handle imbalanced dataset with SMOTE and BalancedRandomForest


#### Health

- [**Stroke Dataset Analysis and Prediction**](https://github.com/MeiChieh/stroke-prediction)  
  `Python` `Scikit-learn` `Data Visualization`
  
  - Feature engineering combining provided feature and external health organization domain knowledge
  - Model selection with imbalanced data suitable metrics and learning curve
- [**Mental Health in the Tech Industry Analysis**](https://github.com/MeiChieh/mental-health-in-tech)  
  `Python` `Pandas` `SQL` `Data Visualization`
  
  - Analyze health data and discover insights through feature aggregation 
  - Use SQL for data wrangling and seaborn for visualization

#### Hobby

- [**European Soccer Matches Time Series Analysis and Prediction**](https://github.com/MeiChieh/european-soccer-matches-prediction)  
  `Python` `Pandas` `DuckDB` `Data Analysis` `Scikit-learn`
  
  - Use feature engineering to generate predictive features and verify with hypothesis testing
  - Construct predictive classifier with logistic regression

- [**Podcast Rating Analysis**](https://github.com/MeiChieh/podcast-rating-analysis)  
  `Python` `Data Analysis` `Pandas` `SQL` `DuckDB` `Scikit-learn`
  
  - Analyzes 2 million review ratings across 100,000 podcasts
  - Identify temporal patterns and distinguish user behaviours across categories

### 🧠 Deep Learning

Advanced machine learning projects using neural networks.

#### Natural Language Processing

- [**Fake News Classifier**](https://github.com/MeiChieh/fake-news-detection)  
  `Python` `PyTorch` `NLTK` `Word2Vec` `BERT` `Longformer` `LIME`
  - Extensive text data cleaning with regex
  - Create heuristic features with sentiment analysis and TF-IDF vectorizer.
  - Fine-tune and compare classification result of transformer models with different maximum token length and cases.
  - Compare result of transformer models and ML models using heuristic features.

#### Computer Vision

- [**Age and Gender Classifier with Face Image**](https://github.com/MeiChieh/face-image-age-and-gender-prediction)  
  `Python` `Resnet` `Squeezenet` `OpenCV` `PyTorch`
  - Real-time object detection
  - Custom dataset training

### 🤖 Generative AI

- [**Turing College Learner RAG Chatbot**](https://github.com/MeiChieh/turing-college-learner-questions-rag-chatbot)  
  `Python` `Streamlit` `FastAPI` `BeautifulSoup` `vectorDB` `Langchain` 
  - Scrape webpage, truncate and save in vectorDB, use dense (cosine similarity) + sparse (BM25) for RAG retrieval
  - Use langchain and streamlit to construct a RAG chatbot that answers learner's questions about the learning platform

## 🛠️ Core Competencies

### Technical Skills

- **Languages**: Python, JavaScript, TypeScript, SQL
- **ML/DL Frameworks**: PyTorch, Scikit-learn, Hugging Face, OpenCV
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Development**: React, Node.js, FastAPI
- **Cloud & DevOps**: Docker, Git
- **AI**: OpenAI SDK, Langchain, Langgraph

### Domain Expertise

- Statistical Analysis & Modeling
- Machine Learning Pipeline Development
- Deep Learning Architecture Design
- Model Deployment
- GenAI Project Development

## 📫 Contact

Feel free to reach out if you have any questions, collaboration ideas or job opportunities !

- Project Repos: [GitHub Repos](https://github.com/MeiChieh?tab=repositories)
- LinkedIn: [Linkedin Page](https://www.linkedin.com/in/mei-chieh-chien-68304798/?trk=opento_sprofile_topcard)
- Email: meichieh.chien@gmail.com
- Portfolio: [GitHub Pages](https://github.com/MeiChieh)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

_This portfolio is continuously updated with new projects and improvements._
