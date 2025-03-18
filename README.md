# 🎭 Feature Selection Analysis for Sentiment Classification

## 📖 Overview
This project evaluates **feature selection methods** (Mutual Information, Chi-Square, and Frequency-Based) for **Naïve Bayes classification** in **sentiment analysis of movie reviews**. We use the **IMDB dataset** to classify reviews as **positive or negative**.

## 🚀 Key Features
- **Text Preprocessing**: Tokenization, stopword removal, stemming.
- **Feature Selection Methods**: Mutual Information, Chi-Square, Frequency-Based.
- **Performance Metrics**: Accuracy, Precision, Recall, F1 Score.
- **Execution Time Profiling**: Using `cProfile` and `pstats`.

---

## 📂 Project Structure
```plaintext
feature-selection-analysis/
│-- 📂 data/                   # Input dataset & preprocessing files
│   │-- IMDB.csv               # Original dataset
│   │-- stopwords.txt          # Stop words list
│   │-- special-chars.txt      # Special characters list
│
│-- 📂 images/                 # Images used in the report
│
│-- 📂 reports/                # Research reports & LaTeX code
│   │-- Report_Team_5.pdf      # Final project report
│   │-- latex_code_Team_5.tex  # LaTeX code for Overleaf report
│
│-- 📂 src/                    # Python source code
│   │-- Code_Team_5.py         # Original project code
│   │-- Time_Analysis_code_Team_5.py  # Code for profiling execution time
│
│-- requirements.txt            # Python dependencies
│-- .gitignore                  # Ignore unnecessary files
│-- README.md                   # Project documentation

## 📖 Report
The full research report can be found **[here](./reports/Report_Team_5.pdf).**

---

## 📊 Key Findings
- **Chi-Square feature selection** performed the best.
- **Mutual Information selection** took the longest execution time.

---

## 👥 Contributors
- **Mounika Gullapalli**
- **Leela Chittoori**

---

## 🎯 Next Steps
- Optimize **Mutual Information execution time**.
- Experiment with **different ML classifiers** (e.g., SVM, Random Forest).
- Expand dataset for **better generalization**.

---

## 🔗 References
- **GitHub Repo**: [Mounika-Gullapalli/feature-selection-analysis](https://github.com/Mounika-Gullapalli/feature-selection-analysis)
- **Git LFS Docs**: [GitHub Large File Storage](https://git-lfs.github.com/)

