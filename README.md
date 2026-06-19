# Explainable Fuzzy Ensemble Genetic Framework for High Precision Crop Recommendation

This repository contains the data and code for an Explainable Fuzzy Ensemble Genetic Framework aimed at providing high-precision crop recommendations. The project leverages machine learning, fuzzy logic, genetic algorithms, and explainable AI techniques (like SHAP and LIME) to build a robust and interpretable agricultural recommendation system.

## Project Structure

*   `data/`: Contains the datasets used for training and testing the models.
    *   `Train Dataset.csv`: The main training dataset.
    *   `SMOTE Test Dataset.csv`: The test dataset, potentially oversampled using SMOTE to handle class imbalance.
*   `notebooks/`: Jupyter notebooks with the core logic, EDA, and model training.
    *   `cleaningcropdataset.ipynb`: Notebook detailing the data cleaning and preprocessing steps.
    *   `projectcrops.ipynb`: The main notebook containing the implementation of the fuzzy ensemble genetic framework and model evaluation.

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/yourusername/crop-recommendation-framework.git
    cd crop-recommendation-framework
    ```

2.  (Optional but recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Navigate to the `notebooks/` directory and start Jupyter to interact with the code:

```bash
cd notebooks
jupyter notebook
```

Open `cleaningcropdataset.ipynb` to view the data preparation steps, or `projectcrops.ipynb` to see the model training, evaluation, and explainability features.

## License

MIT License
