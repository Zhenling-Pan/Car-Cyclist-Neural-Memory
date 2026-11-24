# Car–Cyclist Traffic Hazard Prediction with Neural Memory Retrieval

**Independent PyTorch Research Prototype | Contrastive Learning | Memory Retrieval | Traffic Hazard Prediction**

## Overview

I developed this prototype to explore a research question:

**Can a neural network learn meaningful relationships between traffic situations to retrieve relevant past events and support hazard prediction?**

The scenario involves a driver in an ego car encountering a cyclist. I investigate how neural networks can learn representations of traffic situations and use them to retrieve similar historical events.

The project compares traditional feature-based retrieval, neural embedding-based retrieval, and direct neural network prediction.

## Methodology

### 1. Synthetic Traffic Data

I generated 1,600 synthetic car–cyclist traffic scenarios using five features:

- **Car Speed:** Speed of the ego vehicle.
- **Bicycle Distance:** Distance between the car and cyclist.
- **Cyclist Crossing:** Whether the cyclist crosses the vehicle's path.
- **Visibility:** Whether visibility is reduced.
- **Driver Distraction:** Whether the driver is distracted.

Each scenario has a binary label: **Safe (0)** or **Hazard (1)**, generated using simplified risk rules.

The dataset is divided into 1,200 training samples and 400 test samples.

### 2. Neural Representation Learning

I implemented a neural encoder using PyTorch to transform traffic situations into 8-dimensional embeddings.

The encoder is trained using **Contrastive Learning**, where:

- Situations with the same outcome label are encouraged to have similar embeddings.
- Situations with different outcome labels are encouraged to have more distant embeddings.

This allows the model to learn a representation of traffic situations instead of relying entirely on predefined feature distances.

### 3. Memory Retrieval

I implemented two approaches to retrieve similar historical traffic events:

**Raw-feature KNN:** Retrieves similar situations based on standardized original features.

**Neural-embedding KNN:** Retrieves similar situations using the learned embeddings produced by the neural encoder.

Both methods retrieve the seven nearest historical events and estimate the hazard probability from their labels.

### 4. Prediction and Evaluation

I also trained a separate PyTorch multilayer perceptron (MLP) for direct hazard prediction.

I compared four approaches using Accuracy and F1 Score:

| Method | Accuracy | F1 Score |
|---|---:|---:|
| Majority-class Baseline | 59.5% | 0.000 |
| Raw-feature KNN | 79.0% | 0.747 |
| Neural-embedding KNN | 78.2% | 0.737 |
| Direct PyTorch MLP | 81.0% | 0.768 |

*Results from one experimental run on 400 held-out synthetic traffic scenarios.*

## Findings

In this experiment, the neural embedding-based retrieval approach did not outperform traditional KNN.

This suggests that learning a representation does not automatically improve retrieval performance. The effectiveness of learned embeddings depends on the training objective, the data, and how similarity is defined.

The experiment helped me explore how neural networks can learn relationships between historical events and how different retrieval approaches can be evaluated quantitatively.

## Technologies

Python · PyTorch · NumPy · Scikit-learn · Pandas · Matplotlib

## Notebook

[Car_Cyclist_Neural_Memory_Research.ipynb](Car_Cyclist_Neural_Memory_Research.ipynb)

The notebook includes data generation, neural network training, contrastive representation learning, memory retrieval, model evaluation, and visualizations.

## Limitations and Future Work

This is a simplified research prototype using synthetic data, rather than a validated real-world traffic accident prediction system.

The current implementation uses static traffic scenarios rather than continuous vehicle and cyclist trajectories. The learned similarity is based on synthetic outcome labels and does not establish causal relationships between events.

Additionally, neural retrieval and direct prediction are evaluated separately rather than integrated into a complete memory-augmented prediction model.

Possible future extensions include:

- Incorporating temporal trajectories and behavioral information.
- Exploring more advanced methods for learning relationships between traffic events.
- Integrating retrieved memories into neural prediction models.
- Evaluating prediction performance using realistic traffic simulation data.

---

*This is an independent personal project developed for exploring neural representation learning and memory-based traffic prediction. It is not affiliated with or endorsed by Ericsson.*
