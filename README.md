# ML4Fraud: Machine Learning for E-Commerce and Banking Fraud Detection

**Repository:** [https://github.com/segnig/ml4fraud-ecommerce-banking](https://github.com/segnig/ml4fraud-ecommerce-banking)

## Overview

The **ML4Fraud** initiative represents an integrated machine learning framework that allows for the detection, classification and mathematical interpretation of anomalous (i.e., fraudulent) transactions within two different types of financial industries: e-commerce and traditional banking (i.e., credit card telemetry). The project employs a number of advanced data preprocessing methodologies, mathematically resolves extreme class imbalance, utilizes multiple different machine learning algorithms to accomplish its task, and implements state-of-the-art explainable artificial intelligence models so that the decision-making process behind the algorithms is transparent and trustworthy.

This repository serves as a practical implementation and comparative analysis of fraud detection strategies, making it suitable for academic research, business intelligence benchmarking, and algorithmic auditing applications.

## Goals/Objectives

1. Identify fraudulent activity through all types of transactions; however, given that approximately 1% of all transactions are fraud (and so the ratio of fraud to non-fraud is 1:99), maximizing model Recall (proportion of fraudulent transactions classified as fraudulent) will be the primary goal.
2. Perform complex feature engineering by developing both mathematical and temporal features from the raw time-stamped transaction data (e.g. date-time) and converting the raw timestamp to continuous numerical vectors so that they can be used with any ML classifier.
3. Evaluate and compare multiple ML algorithms including: several linear models as baseline (traditional / parameterized) against a selection of modern ensemble-based models (Random Forest, LightGBM, XGBoost, CatBoost) and one or more foundational Deep Neural Networks.
4. Utilize SHAP (SHapley Additive exPlanations) to explain the reasoning behind a model's decision-making process by decomposing a model output into a set of mathematically-grounded features that can be interpreted as human-readable.

## System Architecture

The ML4Fraud architecture relies on a multi-stage pipeline:
1.  **Data Ingestion & Profiling:** Raw CSV data is ingested, and initial exploratory data analysis (EDA) evaluates feature distributions and missing values.
2.  **Preprocessing & Transformation Engine:** Handles temporal conversions, categorical feature encoding (One-Hot & Label Encoding), and standardization.
3.  **Resampling Logic:** Applies threshold-based spatial modifications (SMOTE Over-sampling / Under-sampling) to balance the target class distribution.
4.  **Modeling Core:** Trains and cross-validates multiple algorithms using `GridSearchCV` for hyperparameter optimization.
5.  **Interpretability Engine:** Calculates Shapley values for the trained ensemble models, exporting global and local interpretation visual plots.

## Project Structure

```text
ml4fraud-ecommerce-banking/
├── notebooks/
│   └── fraud.ipynb                         # Exploratory Data Analysis & initial data profiling
├── models/
│   ├── modelling_credict.ipynb             # Credit card fraud model training pipeline
│   ├── modelling_fraud_data.ipynb          # E-commerce fraud model training pipeline
├── model-explanations/
│   ├── lgbm_fraud_model_explanation.ipynb  # SHAP analysis and visual plots for LightGBM
│   └── rf_credit_model_explanation.ipynb   # SHAP analysis and visual plots for Random Forest
├── src/
│   └── EDA.py                              # Modular Python source code for exploratory queries
├── requirements.txt                        # Global dependency manifest
└── README.md                               # Comprehensive project documentation
```

## Methodology

The methodology flows linearly from raw data to interpretable output:
1.  **Temporal Feature Extraction:** The discrepancy between user signup and transaction events (`time_gap_between_purchase_signup`) is converted from categorical strings to `timedelta` objects and subsequently resolved into continuous numerical values (total seconds). This captures the rapid velocity and automated frequency indicative of fraudulent bots.
2.  **Categorical Encoding Strategy:** Features such as geographic location ('country') are mapped into algorithmic space using both Label Encoding and One-Hot Encoding to empirically determine the optimal representation for tree-based models versus linear structures.
3.  **Cross-Validation:** Robust 5-fold `StratifiedKFold` cross-validation guarantees the algorithms are evaluated across geographically and temporally consistent splits, maximizing generalization.
4.  **Metric Optimization:** Due to severe class imbalance, accuracy is discarded as the primary metric. Instead, the models optimize the **F1-Score**, seeking the harmonic mean between precision and recall.

## Algorithms and Mathematical Formulas

### 1. [Class Imbalance Resolution (SMOTE)](https://www.mdpi.com/2076-3417/13/6/4006)
The Synthetic Minority Over-sampling Technique (SMOTE) is employed to handle massive class imbalances without replicating exact data points. For a minority class sample $x_i \in X_{min}$, SMOTE identifies the $k$-nearest neighbors. A new synthetic sample $x_{new}$ is generated by interpolating between $x_i$ and a randomly chosen neighbor $\hat{x}_i$:

$$ x_{new} = x_i + \lambda (\hat{x}_i - x_i) $$

Where $\lambda$ is a random number between $[0, 1]$. This geometric approach allows classifiers to build robust decision boundaries around the sparse fraud class.

### 2. [Gradient Boosting (LightGBM/XGBoost/CatBoost)](https://www.mdpi.com/2072-6651/15/10/608)
Tree-based gradient boosting models iteratively add weak learners (decision trees) to minimize a predefined loss function. At iteration $t$, the model $F_t(x)$ is updated by adding a new tree $h_t(x)$ that learns the pseudo-residuals of the previous model:

$$ F_t(x) = F_{t-1}(x) + \gamma_t h_t(x) $$

Where $\gamma_t$ is the learning rate. For binary classification (fraud vs. legitimate), the objective often minimizes Log Loss (Binary Cross-Entropy):

$$ L(y, p) = - [y \log(p) + (1 - y) \log(1 - p)] $$

### 3. [Foundational Deep Neural Networks (DNN)](https://www.ncbi.nlm.nih.gov/books/NBK583971/)
A sequential architecture utilizing Dense layers activated by non-linear Rectified Linear Units (ReLU), represented as $f(x) = \max(0, x)$, and terminating in a Sigmoid output function to map the final node to a probability distribution between $[0, 1]$:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

### 4. [Explainable AI: SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
To counter the "black-box" nature of gradient boosting ensembles, SHAP assigns an importance value to each feature for a specific prediction based on cooperative game theory. The Shapley value $\phi_i$ for feature $i$ is calculated by evaluating the model with and without the feature across all subsets $S$ of features:

$$ \phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right] $$

Where $N$ is the set of all features, and $v(S)$ is the prediction value for the subset $S$.

## Implementation Details

*   **Programming Language:** Python 3.10+
*   **Data Processing:** `pandas`, `numpy`
*   **Machine Learning (Traditional & Resampling):** `scikit-learn`, `imbalanced-learn`
*   **Machine Learning (Ensemble Boosting):** `xgboost`, `lightgbm`, `catboost`
*   **Deep Learning:** `tensorflow` / `keras`
*   **Interpretability:** `shap`
*   **Visualization:** `matplotlib`, `seaborn`, `plotly`

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/segnig/ml4fraud-ecommerce-banking.git
    cd ml4fraud-ecommerce-banking
    ```

2.  **Initialize an Isolated Virtual Environment:**
    Creating a controlled Python environment ensures dependency stability:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Dependencies:**
    Install fundamental data science libraries.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The logic flows linearly through interactive, reproducible Jupyter Notebooks:

### 1. Data Profiling
Execute `notebooks/fraud.ipynb` to initiate data profiling, inspect missing values (`NaN`), and review initial spatial distributions and feature anomalies.

### 2. Algorithmic Pipeline Execution
*   **E-commerce Fraud Data:** Run `models/modelling_fraud_data.ipynb`. This incorporates the full training pipeline: data encoding, spatial sampling (SMOTE/Undersampling), ensemble training, cross-validation, and deep learning initialization.
*   **Banking (Credit Card) Data:** Run `models/modelling_credict.ipynb` to evaluate traditional transaction datasets against Logistic Regression, LightGBM, and optimized Random Forest models.

### 3. Interpretability & Reason Code Generation
Navigate to the `model-explanations/` directory. Executing `rf_credit_model_explanation.ipynb` will calculate localized SHAP values and output visual data charts unpacking the logical reasoning of the Random Forest model.

## Example Results

### Credit Card Fraud Detection (Banking Sector)
An optimized **Random Forest Classifier** emerged as the strongest candidate on standard banking metrics, achieving high detection rates with minimal false positives:
*   **Accuracy:** 99.96%
*   **ROC-AUC Score:** 89.28%
*   **Precision:** 97.47% (Highly reliable positive identifications)
*   **Recall:** 78.57% (Caught ~80% of all synthesized and real fraud cases)
*   **F1-Score:** 87.01% (Harmonic mean of precision and recall)

### E-Commerce Fraud Detection (Telemetry Sector)
A **LightGBM Classifier**, heavily optimized via multidimensional Grid Search (`depth=4`, `learning_rate=0.01`), demonstrated the most robust detection capabilities, yielding a stabilized F1-Score of **~69.8%** on heavily imbalanced, structurally un-sampled data.

### Global Feature Importance
SHAP summary plots indicate that mathematical derivatives of user behavior—specifically the temporal proximity between account creation and initial transaction—exhibit the highest Shapley correlation values $\phi_i$ toward fraudulent classifications.

## Applications

*   **Financial Risk Management:** Algorithms providing ~97% precision effectively act as primary filters for real-time authorization logic, minimizing transactional impedance for legitimate consumers.
*   **Auditing and Compliance (XAI):** Utilizing algorithmic predictions mapped against SHAP Force Plots allows compliance officers to immediately view a mathematically sound "Reason Code" for declined transactions, aiding in regulatory compliance.
*   **Dynamic Network Architecture Analysis:** Real-time classification of bot networks operating on temporal frequencies (high volume, zero-day account creation).

## References and Learning Resources

The theoretical concepts and algorithms applied in this repository are based on the following established academic and technical foundations:

*   **SMOTE (Synthetic Minority Over-sampling Technique):** 
    *   Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*. [Read Paper (ArXiv)](https://arxiv.org/abs/1106.1813)
*   **Gradient Boosting Frameworks:**
    *   Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *ACM SIGKDD*. [Read Paper (ArXiv)](https://arxiv.org/abs/1603.02754)
    *   Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*. [Read Paper (NeurIPS)](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
*   **SHAP and Explainable AI:**
    *   Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*. [Read Paper (NeurIPS)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
    *   *Official SHAP Documentation*: [shap.readthedocs.io](https://shap.readthedocs.io/en/latest/)
*   **Deep Learning Fundamentals:**
    *   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Online Book](https://www.deeplearningbook.org/)

## Limitations and Future Improvements

*   **Feature Expansion:** The dataset fundamentally relies on standard behavioral metrics. Future iterations should incorporate device fingerprinting data (e.g., hashed IP clusters, MAC addresses) to build stronger structural barriers against chargeback rings.
*   **Recurrent Architectural Enhancements:** While Sequential Neural Networks provide baseline estimations, transitioning the deep learning architecture toward LSTMs (Long Short-Term Memory networks) may better capture temporal series degradation in multi-transaction user journeys.

## Contributing

Contributions are welcomed. Please fork the repository, create a dedicated feature branch, and submit a detailed Pull Request outlining your algorithmic or preprocessing adjustments. Ensure any new models introduced execute standard 5-fold cross-validation.

## Contact & Support

Questions? Suggestions? Reach out!

**Segni**
Email: [segnigirma11@gmail.com](mailto:segnigirma11@gmail.com)
GitHub: [https://github.com/segnig](https://github.com/segnig)

## License

This project is licensed under the MIT License — see the LICENSE file for details.
