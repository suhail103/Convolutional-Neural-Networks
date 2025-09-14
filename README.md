# Natural Language Processing Lab 2: Convolutional Neural Networks

This project explores the effect of **architectural depth** on Convolutional Neural Networks (CNNs) for text classification.  
Using text from the *Harry Potter* series, we compare a **single-layer CNN (Base Model)** and a **two-layer CNN (Deep Model)** to determine how network complexity impacts performance.

---

## 🚀 Project Overview
The primary task was to predict **which Harry Potter book a given text excerpt came from**.  
The pipeline involves:

1. **Data Preprocessing**
   - Clean and tokenize raw text.
   - Segment the corpus into 200-word sequences.
   - Remove punctuation and convert to lowercase.

2. **Word Embedding**
   - A custom **Word2Vec encoder** was trained on the dataset.
   - The embedding layer was frozen during CNN training for fair comparison.

3. **CNN Architectures**
   - **BaseCNN:** Single convolutional block with filter sizes `[3, 4, 5]`.
   - **DeepCNN:** Two stacked convolutional blocks to increase receptive field and complexity.

4. **Training & Evaluation**
   - Dataset split: 80% training / 20% testing.
   - Models trained for 15 epochs using Adam optimizer and CrossEntropy loss.
   - Performance evaluated on accuracy and loss metrics.

---

## 📊 Key Findings
| Model          | Avg. Train Accuracy | Avg. Test Accuracy |
|----------------|---------------------|--------------------|
| Base CNN (1 Layer) | 94.72%              | **67.30%**          |
| Deep CNN (2 Layers) | 98.73%              | 66.27%              |

- The **single-layer CNN generalized better**, achieving higher test accuracy.
- The **two-layer CNN overfit**, with a large gap between training and test accuracy.
- Additional depth only improved memorization, not real-world performance.

---

## 🗂 Repository Structure
