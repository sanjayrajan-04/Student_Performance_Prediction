# Student Performance Prediction and Analysis

This project is a **machine learning web application** that predicts student performance (grades A--F) based on academic and personal factors.
It combines data analysis, machine learning, and Flask web development to deliver an interactive tool for analyzing and predicting outcomes.


## Project Overview

1) The dataset ("Student_performance_data.csv") contains academic records and related features.

2) Data preprocessing, training, and evaluation are handled in "train_model.py" and the Jupyter Notebook ("Student Performance.ipynb").

3) A **Decision Tree Classifier** is trained and saved as "student_performance_model.pkl".

4) The Flask app (app.py) loads the trained model and serves a simple web interface for prediction.

5) Visuals (Grade.PNG) are provided for grade distribution.

-

## Tech Stack

1) Python (Core Language)

2) Flask(Web Framework)

3) scikit-learn(Machine Learning)

4) pandas / numpy (Data Handling)

5) matplotlib / seaborn (Visualization)

6) joblib (Model Serialization)



## Project Structure

    Student Performance Prediction

    * app.py                        # Flask web app
    * train_model.py                # Script to train & save the ML model
    * Student Performance.ipynb     # Jupyter notebook with data analysis
    * Student_performance_data.csv  # Dataset
    * student_performance_model.pkl # Trained ML model
    * Grade.PNG                     # Sample visualization
    * templates/                    # HTML templates for Flask
    * .venv/                        # Virtual environment (ignore in git)


## Installation & Setup

1.  Clone the repository

2.  Create and activate a virtual environment

3.  Install dependencies

4.  Train the model (if needed)

5.  Run the Flask app

6.  Open in browser


## Future Enhancements

-   Support for more ML models (Random Forest, Gradient Boosting).\
-   Deploy on cloud (Heroku / Render / AWS).\
-   Add interactive dashboard for visual analytics.


## Requirements

The project requires the following Python libraries:

-   flask\
-   numpy\
-   pandas\
-   scikit-learn==1.0.2\
-   joblib\
-   matplotlib\
-   seaborn


## Conclusion

This project demonstrates how machine learning can be applied to education to predict student performance and provide actionable insights. By leveraging a Decision Tree Classifier, the model achieves ~84% validation accuracy in classifying students grades (A-F).The interactive Flask web application makes the system practical for educators and institutions, enabling them to quickly input student data and receive predictions. Such tools can help in identifying at-risk students early, improving academic planning, and supporting data-driven decision-making in education.

