BTP Decision-Making Helper Dashboard

Project Overview

This project provides a dashboard for decision-making assistance, integrating a machine learning model with a FastAPI backend and a Streamlit frontend. Users can interact with the dashboard to explore model predictions in a user-friendly interface.

-Folder Structure
*BTP_Decision_making_helper/
-backend/ (FastAPI backend code)
-frontend/ (Streamlit frontend code)
-models/ (Pre-trained ML models)
-Data_and_functions/ (Utility functions and data processing)
-test.py (Testing scripts)
*requirements.txt (Python dependencies)
*README.md (Project documentation)

-Dashboard Setup and Usage

1.Clone the repository
git clone https://github.com/Chotesharmaji/BTP_Strategy.git
cd BTP_Strategy

2.Download the pre-trained model
Download the model from https://drive.google.com/file/d/1AhZmH-M-LjsABef-HcBwuF0VRITUjzIO/view?usp=sharing
and move it to the models/ directory in the project.

3.Set up a Python virtual environment
Create and activate a virtual environment
Install dependencies
pip install -r requirements.txt

4.Run the FastAPI backend
From the project root:
uvicorn main:app --reload
Backend URL: http://127.0.0.1:8000
API docs (Swagger UI): http://127.0.0.1:8000/docs

5.Run the Streamlit frontend
Navigate to the frontend folder and start the dashboard:
cd frontend
streamlit run app.py

6.The dashboard should open automatically in your browser.
Enjoy!
You can now interact with the dashboard and explore the model predictions.