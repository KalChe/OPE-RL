# ⚕️ {WIP} - Work-in-progress

## 🧠 Features

* ✅ **Synthetic patient simulator** modeling pain, withdrawal, and behavioral dose trends
* ✅ **MDP formulation** with interpretable clinical states, actions, and reward modeling
* ✅ **Offline RL agents**:

  * **CQL** (Conservative Q-Learning)
  * **BCQ** (Batch-Constrained Q-Learning)
* ✅ **Offline Policy Evaluation (OPE)**:

  * Importance Sampling
  * Fitted Q-Evaluation
  * Model-Based Rollouts
* ✅ **COBS**: Conservative policy selection with bootstrapped confidence bounds
* ✅ **Deployment interface** for personalized tapering recommendations

---

## 🏗️ Pipeline Components

### 🔬 1. Synthetic Dataset

* Simulates 500 patients over 52 weeks
* Includes demographics, dose, pain, withdrawal, and behavioral noise
* Prepares behavior policy data for BCQ

### ⚙️ 2. MDP Setup

* **States**: `[dose, pain, withdrawal, demographics, history]`
* **Actions**: 21 discrete tapering options (-50 to +10 MME)
* **Rewards**: Trade-off between pain, withdrawal, and tapering goals
* **Transitions**: Pharmacodynamic-informed updates

### 🤖 3. Offline RL Algorithms

* **CQL** with Q-function regularization
* **BCQ** with behavior cloning filter

### 📈 4. Offline Policy Evaluation

* On-policy and off-policy estimators
* Confidence-aware model-based rollouts

### 🛡️ 5. COBS Policy Selection

* Bootstrapped OPE values
* Selects policies with high-confidence performance guarantees

### 💻 6. Clinical Decision Support Interface

**TODO**
---

## ▶️ How to Use

**TODO**

---

* The script automatically:

  * Generates synthetic data
  * Trains CQL and BCQ agents
  * Evaluates using OPE and COBS
  * Outputs policy recommendations

---

## 📊 Visualizations

The script includes built-in plots for:

* Patient trajectories (dose, pain, withdrawal)
* CQL vs BCQ policy performance
* Confidence intervals for COBS selection

---

## 📬 Contact

Built by **Kalyan Cherukuri**
Email: \[kcherukuri@imsa.edu]
Feel free to reach out for questions, ideas, or collaboration.
