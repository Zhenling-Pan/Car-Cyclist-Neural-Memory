# Car–Cyclist Hazard Prediction with Neural Memory Retrieval

This project investigates whether learned neural embeddings improve the retrieval of similar traffic scenarios compared with conventional feature-based retrieval. The experiment uses simulated encounters between a car and a cyclist, with PyTorch for representation learning and classification.

## Experimental setup

The dataset consists of 1,600 synthetic traffic scenarios described by five features:

- Vehicle speed
- Distance to the cyclist
- Cyclist crossing status
- Reduced visibility
- Driver distraction

Binary hazard labels are generated using simplified risk rules. The dataset is split into 1,200 training samples and 400 test samples.

Four methods are evaluated:

- Majority-class baseline
- KNN using standardized input features
- KNN using learned neural embeddings
- Direct classification using an MLP

The neural encoder maps each scenario to an 8-dimensional embedding. Contrastive learning is used to reduce the distance between samples with matching labels and increase the separation between samples with different labels.

Both retrieval methods use seven nearest neighbours.

## Results

| Method | Accuracy | F1 score |
|---|---:|---:|
| Majority-class baseline | 59.5% | 0.000 |
| Raw-feature KNN | 79.0% | 0.747 |
| Neural-embedding KNN | 78.2% | 0.737 |
| Direct MLP | 81.0% | 0.768 |

Results are based on a single experimental run with 400 held-out samples. The learned embeddings did not improve KNN performance in this experiment. Direct classification achieved the highest accuracy and F1 score. One possible explanation is that the original features already capture much of the information used to generate the hazard labels. A different training objective or a more complex dataset may produce different results.

## Running the notebook

The full experiment is available in [Car_Cyclist_Neural_Memory_Research.ipynb](Car_Cyclist_Neural_Memory_Research.ipynb).

Dependencies:

- PyTorch
- NumPy
- scikit-learn
- pandas
- Matplotlib

Open the notebook in Jupyter or Google Colab and run the cells in order. The dataset is generated within the notebook.

## Limitations

The experiment uses static, synthetic traffic scenarios rather than recorded vehicle and cyclist trajectories. The results therefore do not establish real-world accident prediction performance. The retrieval and classification models are evaluated separately. A natural next step is to incorporate retrieved embeddings into the prediction model and test whether they provide additional information.

---

Independent personal project. Not affiliated with Ericsson.
