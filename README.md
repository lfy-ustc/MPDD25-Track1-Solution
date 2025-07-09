# MM 2025 MPDD Challenge - Track1 MPDD-Elderly Solution

This repository contains our solution for **Track 1: MPDD-Elderly Task** in the **MM 2025 MPDD Challenge**.

---

## 🔧 Environment

The code was developed and tested under the following environment:

- Python 3.10.0
- PyTorch 2.3.0
- scikit-learn 1.5.1
- pandas 2.2.2

We recommend setting up the environment using **conda** and the provided `requirements.txt` file:

```bash
conda create -n mpdd python=3.10 -y
conda activate mpdd
pip install -r requirements.txt
```

---

## 📦 Reproducing Results

To reproduce the results:

1. Modify the `DATA_ROOTPATH` in the relevant `scripts/test_*.sh` script to match the path of your dataset.
2. Run the script:

```bash
bash scripts/test.sh
```

Predicted results will be saved according to the script's configuration.

---

## 🏋️‍♂️ Training

To train the model:

1. Modify the `DATA_ROOTPATH` in the corresponding `scripts/train_*.sh` script.
2. Run the script with:

```bash
bash scripts/Track1/train_1s_binary.sh
```

> You can switch to other training scripts depending on the task configuration (e.g., 1s/5s, binary/ternary/quinary).

---

## ⚠️ Important Note on Performance Fluctuation

A key observation during our experiments is the **significant variance in model performance** across different training runs. We believe this is attributable to several factors:

* **Limited Dataset Size:** The dataset is not very large, making the model susceptible to overfitting and sensitive to the specific train/validation split. This issue is especially pronounced in the 3-class and 5-class classification scenarios, where the limited number of samples per class means that the correct or incorrect classification of just **a single sample** can cause a substantial swing in the overall performance metrics.
* **Individual Variability:** The dataset is derived from human subjects, who inherently possess wide-ranging differences (e.g., age, health conditions, task execution). This natural biological variation acts as a source of noise, leading to significant variance in the data.

**What to Expect:**

* **High Variance:** Even when using a **fixed random seed**, you may observe noticeable fluctuations in performance metrics (e.g., Accuracy, F1-score) from one run to another. The variance can be quite large.
* **"Lucky" Seeds:** Achieving a high-performance result can be challenging and may depend on a favorable random initialization or data shuffling. Exceptional results are possible but may not be consistently reproducible in every single run.

We encourage users to run the training process multiple times to get a better sense of the average performance and its standard deviation, rather than relying on the outcome of a single run.

---

## 📬 Contact

If you have any questions or encounter any issues, feel free to open an issue or reach out to us.

