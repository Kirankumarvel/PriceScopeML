
# 🏡 PriceScope – Zooming into the Truth Behind Housing Prices

> A deep dive into regression evaluation, bias detection, and model explainability using Random Forest and the California Housing Dataset.

---

## 📌 Project Summary

**PriceScope** is a machine learning mini-project that goes beyond just fitting a model. It unpacks the truth hidden within evaluation metrics and residuals—giving you a clear understanding of:

- Where models perform well or poorly
- How skewed distributions and clipped target values affect performance
- Why R² alone doesn't tell the whole story

This hands-on analysis uses Python, `scikit-learn`, `matplotlib`, and `pandas`, focusing on building **interpretable** and **trustworthy** regression models.

---

## 🚀 Key Highlights

- ✅ Trained a Random Forest Regression model on the California Housing dataset
- 📊 Visualized and interpreted metrics like MAE, MSE, RMSE, and R²
- 📉 Analyzed residuals to detect bias in price predictions
- 🔎 Uncovered the effect of skewed and clipped data
- 🌲 Interpreted feature importance and explored model explainability

---

## 📁 Directory Structure

```
PriceScope/
├── 📄 README.md               # Project overview
├── 📄 price_scope.py          # Main file
├── 📄 data_info.txt           # Dataset description (optional)
└── 📊 results/                # Plots and residual analysis visuals
```

---

## 🔧 Requirements

Make sure to install the following:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## 🧪 How to Run

1. Clone the repository or copy the notebook.
2. Run each cell sequentially in the Jupyter Notebook: `price_scope.ipynb`

---

## 📈 Evaluation Metrics Covered

| Metric  | What it Tells You                         |
|---------|-------------------------------------------|
| MAE     | Average absolute prediction error         |
| MSE     | Penalizes large errors more than MAE      |
| RMSE    | Square root of MSE, easier to interpret in units |
| R² Score| How much variance in the target is explained |

---

## 🔍 Hidden Gems Uncovered

- Why visualizing residuals is as important as looking at MAE/RMSE
- How clipped values (e.g., $500K cap) skew model performance
- The limited utility of R² in nonlinear, skewed datasets
- Feature importance is cool, but beware of feature correlation bias

---

## 🎯 Final Takeaways

- Models can perform well on average, but still fail at the extremes
- Always check residuals sorted by actual values
- Interpret metrics in context, especially with domain knowledge

---

## 📚 Tags

`machine-learning`, `regression`, `random-forest`, `model-evaluation`, `feature-importance`, `bias-detection`, `data-visualization`, `real-estate-ml`, `california-housing`, `skewness`, `residual-analysis`

---

## 🧠 Author

Kumar  
Python Explorer | Machine Learning Enthusiast

---

## 💡 License

MIT – Feel free to use, modify, and build on this!

