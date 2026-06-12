# 🎧 Spotify Musical Landscape Explorer

A dimensionality reduction project that visualizes Spotify audio features using PCA and displays the musical landscape in an interactive Streamlit dashboard.

---

## 🚀 Features

- Upload any Spotify CSV dataset
- PCA Dimensionality Reduction (10 features → 2D)
- Feature Correlation Heatmap
- Scree Plot with Cumulative Variance
- Static 2D Musical Landscape
- Interactive 2D Visualization (hover for track info)
- **Genre Comparison** — radar chart, bar chart, PCA scatter, and stats table side by side
- Spotify-styled dark UI

---

## 📊 Dataset

Download from Kaggle:
👉 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Upload through the app sidebar after launching.

---

## 🛠️ Tech Stack

Python · Streamlit · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Plotly

---

## ▶️ Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/spotify-landscape-explorer.git
cd spotify-landscape-explorer

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
spotify-landscape-explorer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Files excluded from git
```
