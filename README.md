# Ensemble Learning for Complex Regression Modeling on Bike Share Data

## 📖 Project Overview

This project investigates the application of **ensemble learning techniques** to a real-world regression problem, forecasting hourly bike rental demand using the **UCI Bike Sharing Dataset**.  
The analysis begins with single baseline models and progressively evolves through advanced ensemble strategies that tackle both **bias** and **variance** in predictive modeling.

Three core ensemble methods are explored:

1. **Bagging (Bootstrap Aggregation):** focuses on variance reduction by averaging multiple high-variance models.  
2. **Boosting (Gradient Boosting Regressor):** targets bias reduction through sequential error correction.  
3. **Stacking (Meta-Ensemble):** combines multiple diverse learners (KNN, Bagging, Gradient Boosting) using a meta-learner (Ridge Regression) for optimal generalization.

Beyond raw predictive accuracy, the project emphasizes **bias–variance trade-off interpretation**, **cross-validation**, and **residual analysis**. Through a blend of statistical metrics and visualization, the study builds a transparent and theoretically grounded comparison of ensemble behaviors.

##  📁 Repository Structure

```
EnsembleLearning/
│
├── main.ipynb
├── dataset/
│ └── hour.csv
└── README.md
```

- **main.ipynb** — The Jupyter Notebook containing all implementation code, visualizations, cross-validation analysis, and final conclusions.  
- **dataset/hour.csv** — The UCI Bike Sharing Demand dataset (hourly records).  
- **README.md** — Documentation detailing objectives, methods, and execution instructions.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

To install the required libraries, run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
## How to Run This Project
1. **Clone the Repository**
    ```bash
    git clone https://github.com/Charukhesh/EnsembleLearning.git
    cd EnsembleLearning
    ```

2. **Dataset**
    Ensure hour.csv is located inside the dataset/ folder.

3. **Open and Run the Notebook**
    - Open `main.ipynb` in [Jupyter Notebook](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/).
    - Execute all cells sequentially to reproduce preprocessing, model training, ensemble evaluations, and final visual analyses.

## Citation

**Dataset Source:**
Fanaee-T, Hadi, and Gama, João. “Event labeling combining ensemble detectors and background knowledge.” Progress in Artificial Intelligence, 2013 - [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

