# Social Circles: Community Analysis and Link Prediction Using Facebook100

## Project Member
- Khushi Patel

## Project Dataset: Basic Dataset Statistics
The dataset used in this project is the Facebook100 dataset, sourced from the Stanford Network Analysis Project (SNAP). This dataset is available at: Facebook100 Dataset. It contains anonymized user interactions and friendships within various college networks, represented as an undirected graph.

### Basic Statistics
- **Total nodes (users):** 4,039
- **Total edges (connections):** 88,234
- **Average degree (average number of connections per user):** 43.69

## Methodology
We will use a supervised link prediction approach. This involves predicting potential friendships (edges) in the network based on existing structural properties.

### Suggested Approach
1. **Data Preprocessing:** 
   - Clean and preprocess the network data, including removing isolated nodes and normalizing features such as degree centrality and clustering coefficients.
   
2. **Feature Engineering:** 
   - Generate features based on graph properties, such as common neighbors, Jaccard coefficient, and preferential attachment.
   
3. **Model Selection:** 
   - Train machine learning models (e.g., logistic regression, random forest) using the engineered features to predict the existence of links.
   
4. **Evaluation:** 
   - Use metrics like AUC-ROC and F1 score to evaluate the link prediction performance.

## Installation
To run this project, you will need to install the following dependencies:
- Python 3.x
- NetworkX
- Scikit-learn
- Pandas
- NumPy


## Usage
1. **Download the dataset** from Facebook100 Dataset.
2. **Run the preprocessing script** to clean and normalize the data.
3. **Generate features** using the feature engineering script.
4. **Train the model** using the model selection script.
5. **Evaluate the model** using the evaluation script.

## Results
The results of the link prediction will be evaluated using AUC-ROC and F1 score metrics. Detailed results and analysis will be provided in the final report.

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

