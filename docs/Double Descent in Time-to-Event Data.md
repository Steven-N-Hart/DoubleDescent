# **Proposal for a Scientific Manuscript: Investigating the Double Descent Phenomenon in Time-to-Event Analysis Through Synthetic Data Stress-Testing**

## **1\. Introduction and Rationale**

### **1.1 The Paradigm Shift in Statistical Learning Theory**

For the majority of the twentieth century, statistical learning theory was governed by a singular, pervasive dogma: the bias-variance trade-off. This classical perspective posits that generalization error follows a U-shaped curve with respect to model complexity. As a model's capacity increases—typically measured by the number of trainable parameters—the bias decreases because the model becomes flexible enough to capture the underlying signal. Simultaneously, however, the variance increases as the model gains the capacity to fit the stochastic noise inherent in the training data. The "sweet spot" for generalization was theoretically located at the trough of this curve, balancing underfitting (high bias) and overfitting (high variance).  
This theoretical framework provided the justification for parsimony in model design, motivating the use of dimensionality reduction, feature selection, and strict regularization (e.g., AIC, BIC, Lasso) to keep the number of parameters p significantly smaller than the sample size n. Yet, the empirical reality of modern deep learning has starkly contradicted this classical wisdom. Neural networks with parameters numbering in the millions or billions—far exceeding the number of training examples (p \\gg n)—regularly achieve state-of-the-art performance without catastrophic overfitting. This paradox led to the identification of the **"Double Descent"** phenomenon.  
The double descent curve subsumes the classical U-shaped curve into a broader framework. It describes two distinct regimes separated by an **interpolation threshold** (where p \\approx n):

1. **The Classical Regime (Under-parameterized):** Here, the test error behaves as expected, decreasing as the model learns and then increasing as it begins to overfit.  
2. **The Critical Regime (Interpolation Threshold):** As p \\to n, the model has just enough capacity to fit the training data perfectly (zero training error). However, because there is effectively only one (or very few) solutions that pass through every noisy data point, the model is forced into a highly volatile state with massive weight norms and extreme sensitivity to noise. This results in a sharp spike in test error.  
3. **The Modern Regime (Over-parameterized):** As capacity increases further (p \\gg n), the number of possible solutions that fit the training data becomes infinite. Among these, optimization algorithms like Stochastic Gradient Descent (SGD) implicitly select the solution with the minimum norm (e.g., smoothest function). Consequently, the test error descends a second time, often reaching levels lower than the classical minimum.

While this phenomenon has been extensively mapped in computer vision (using Convolutional Neural Networks) and natural language processing (using Transformers), its manifestation in **Survival Analysis (Time-to-Event Analysis)** remains critically underexplored. This gap is significant because survival analysis is the cornerstone of prognostic modeling in medicine, where decisions regarding patient care rely on risk estimates derived from complex, often high-dimensional data.

### **1.2 The Unique Landscape of Survival Analysis**

Survival analysis differs fundamentally from the regression and classification tasks where double descent was discovered. The primary distinction is **censoring**: for a subset of subjects, the event of interest (e.g., death, device failure) is not observed; we only know that the event had not occurred up to a certain time C\_i. This turns the target variable into a pair (Y\_i, \\delta\_i), where Y\_i \= \\min(T\_i, C\_i) and \\delta\_i is the event indicator.  
This data structure necessitates specialized loss functions, such as the **Cox Partial Likelihood**, which depends on the rank ordering of subjects within "risk sets" rather than the absolute difference between predicted and actual values. The geometry of the Cox loss is unique; for perfectly separable data (which becomes likely in the over-parameterized regime), the optimal weights may diverge to infinity to maximize the likelihood, creating a loss landscape that lacks a finite global minimum.  
Recent theoretical explorations by Liu et al. (2025) suggest that while some continuous-time survival models exhibit a "mild" second descent, the overfitting observed in the critical regime is not "benign." Their findings indicate that generalization loss in over-parameterized survival models often remains higher than the classical minimum, challenging the "bigger is better" philosophy of deep learning when applied to medical data. Furthermore, the definition of "interpolation" in survival analysis is ambiguous. Does it mean predicting the exact event time (impossible for censored data) or achieving a concordance index of 1.0?

### **1.3 The Necessity of Synthetic Data Stress-Testing**

Investigating double descent in survival analysis using real-world clinical datasets (e.g., METABRIC, SUPPORT, SEER) is fraught with limitations. Real-world datasets have fixed sample sizes and fixed feature sets, preventing the granular sweep of the p/n ratio required to visualize the full double descent curve. Moreover, real data contains unknown noise structures, unmeasured confounders, and informative censoring that cannot be disentangled from the model's learning dynamics.  
To scientifically isolate the drivers of double descent in survival models, we must utilize **synthetic data generation**. A generative framework allows us to act as the "oracle," defining the ground truth hazard functions and systematically varying data characteristics that are prevalent in healthcare but often ignored in theoretical ML papers:

* **Skewed Distributions:** Clinical biomarkers (e.g., bilirubin, CRP) often follow log-normal or mixture distributions with heavy tails, acting as leverage points that may destabilize interpolation.  
* **High Cardinality:** Categorical variables (e.g., hospital codes, zip codes) introduce extreme sparsity when one-hot encoded, effectively pushing the dimensionality p into the over-parameterized regime even for small networks.  
* **Class Imbalance (Heavy Censoring):** In many medical studies (e.g., screening for rare diseases), the event rate is low (\<5%). This imbalance reduces the "effective" sample size (N\_{eff} \= \\text{number of events}), potentially shifting the interpolation threshold from p \\approx n to p \\approx N\_{eff}.

### **1.4 Proposal Objectives**

This manuscript proposes a comprehensive simulation study to map the double descent landscape in deep survival models. We aim to answer the following research questions:

1. **Existence and Shape:** Does the double descent phenomenon exist in deep Cox models (DeepSurv), and is the "modern regime" actually superior to the classical regime for survival tasks?  
2. **Distributional Sensitivity:** How do non-Gaussian covariates (skewed and categorical) modify the width and amplitude of the critical interval?  
3. **The Censoring Shift:** Does the interpolation peak shift as a function of the censoring rate, confirming the hypothesis that effective sample size drives the phenomenon?  
4. **Metric Divergence:** Do rank-based metrics (C-index) mask the overfitting catastrophe that calibration metrics (Integrated Brier Score) reveal?

## **2\. Theoretical Framework and Literature Review**

### **2.1 The Mechanics of Double Descent**

The theoretical underpinning of double descent lies in the bias-variance decomposition of the generalization error. In the under-parameterized regime (p \< n), the model class \\mathcal{F} is too small to fit the noise \\epsilon. As p increases, bias reduces. At the interpolation threshold (p \\approx n), there exists a unique (or very small set of) parameter configurations \\theta that satisfy f\_\\theta(x\_i) \= y\_i for all i=1 \\dots n. Because the training data includes noise y\_i \= f^\*(x\_i) \+ \\epsilon\_i, forcing the model to interpolate y\_i requires the function to oscillate wildly between data points. This results in a variance explosion:  
At p \\approx n, \\text{Variance} \\to \\infty (or a very large value).  
In the over-parameterized regime (p \\gg n), the condition f\_\\theta(x\_i) \= y\_i defines a manifold of solutions. Optimization algorithms like SGD, initialized with small random weights, tend to converge to the solution on this manifold closest to the initialization. This effectively minimizes the parameter norm ||\\theta||\_2. This implicit regularization leads to a "minimum norm solution" that is smoother than the unique solution at the threshold, causing variance to decrease—the second descent.

### **2.2 Survival Loss Geometry and Interpolation**

Most deep survival models, including **DeepSurv** , optimize the Cox partial likelihood. Let R\_i be the risk set at time t\_i. The loss is:  
Here, h\_\\theta(x) is the predicted log-risk. Unlike Mean Squared Error (MSE), which is convex for linear models and has a minimum at 0, the Cox loss depends on the *relative* values of h\_\\theta(x). If the data is separable (i.e., we can perfectly order the risks such that subjects with events always have higher risk scores than those surviving longer), the loss can be driven towards \-\\infty by scaling h\_\\theta(x) \\to \\infty.  
**The Finite-Norm Constraint:** Theoretical work by Liu et al. (2025) introduces the concept of "finite-norm interpolation." They argue that a sharp double descent peak is observed only when the model can interpolate the data with finite weights. If interpolation requires infinite weights (as is often the case with logistic/Cox loss on separable data), the "valley" of low test error in the over-parameterized regime may not be accessible or stable. This suggests that double descent in survival analysis might be "dampened" or that the critical peak might be wider and more destructive.

### **2.3 The "Effective" Sample Size Hypothesis**

A critical theoretical contribution of this proposal is the examination of **Effective Sample Size**. In regression, n is simply the number of rows. In survival analysis, censored observations provide only partial information (T \> C). The statistical power of a survival study is determined primarily by the number of events E, not n. We hypothesize that the interpolation threshold for survival models is governed by the ratio p / E, not p / n. If true, this has profound implications: a deep learning model might enter the dangerous "critical regime" much earlier than expected if the dataset is heavily censored. For example, a dataset with 10,000 patients but only 100 events might exhibit overfitting peaks when the model has only \\approx 100 parameters, not 10,000.

### **2.4 Data Types as Complexity Modifiers**

The "complexity" of the learning task is not just a function of sample size, but of the data manifold's geometry.

* **Skewed Covariates (Log-Normal):** Medical costs, hospital length of stay, and biomarker concentrations often follow log-normal distributions. These features contain "outliers" in the tail that are valid data points, not artifacts. A model attempting to interpolate these points in the critical regime must stretch its decision boundary significantly, potentially exacerbating the variance spike.  
* **High Cardinality:** Categorical variables with thousands of levels (e.g., ZIP codes) create a sparse feature matrix. When One-Hot Encoded, the input dimension d becomes massive. This forces the model to learn from very few examples per category. We hypothesize that this sparsity induces a form of "lazy learning" where the model memorizes the few training examples in each category bin, leading to poor generalization in the critical regime but potentially stable behavior in the ultra-wide regime due to the orthogonality of the features.

## **3\. Methodology: A Robust Synthetic Data Framework**

To systematically test these hypotheses, we require a generative engine capable of simulating survival times T conditional on covariates X, while strictly controlling the marginal distributions of X and the dependency structure. We adopt the **Inverse Transform Sampling** approach coupled with **Gaussian Copulas** for dependency management.

### **3.1 The Generative Pipeline**

The simulation process is divided into three stages: Covariate Generation, Event Time Simulation, and Censoring Application.

#### **Stage 1: Covariate Generation with Copulas**

Real-world data features are rarely independent. We use a Gaussian Copula to induce correlation structure between variables of different types (continuous and categorical).

1. **Latent Generation:** Generate a matrix \\mathbf{Z} \\in \\mathbb{R}^{n \\times d} from a multivariate normal distribution \\mathcal{N}(0, \\boldsymbol{\\Sigma}), where \\boldsymbol{\\Sigma} defines the correlation structure.  
2. **Marginal Transformation:**  
   * **Normal Variables:** X\_j \= Z\_j.  
   * **Skewed Variables:** X\_j \= \\exp(Z\_j) (Log-Normal transformation).  
   * **Categorical Variables:** X\_j \= \\text{QuantileBinning}(\\Phi(Z\_j), \\text{bins}=K), where \\Phi is the standard normal CDF and K is the cardinality. This preserves the rank correlation induced by the copula while creating discrete categories.

#### **Stage 2: Event Time Simulation (Inverse Transform)**

We assume a Proportional Hazards (PH) framework. The hazard function for subject i is:  
where \\eta\_i \= f(\\mathbf{x}\_i) is the linear (or non-linear) predictor. We use a **Weibull baseline hazard** h\_0(t) \= \\lambda \\nu t^{\\nu-1} because it is flexible and allows for a closed-form inverse. The cumulative hazard is H(t | \\mathbf{x}\_i) \= \\lambda t^\\nu \\exp(\\eta\_i). The survival function is S(t | \\mathbf{x}\_i) \= \\exp(-H(t | \\mathbf{x}\_i)).  
To generate the true event time T\_i:

1. Sample u\_i \\sim \\text{Uniform}(0, 1).  
2. Invert the survival function: S(T\_i | \\mathbf{x}\_i) \= u\_i.  
3. Solving for T\_i: This method allows us to generate exact event times conditional on the simulated covariates.

#### **Stage 3: Censoring Mechanism**

To generate observed data (Y, \\delta), we simulate censoring times C\_i from a distribution P(C). To study the effect of censoring rate, we assume C\_i \\sim \\text{Exponential}(\\lambda\_c) (independent censoring). We calibrate \\lambda\_c iteratively to achieve target censoring rates (e.g., 20%, 50%, 90%).  
.

### **3.2 Specific Data Distribution Scenarios**

We define four specific datasets to stress-test the double descent hypothesis.  
**Table 1: Synthetic Data Scenarios**

| Scenario | Covariate Type | Distribution Details | Complexity Factor | Hypothesis |
| :---- | :---- | :---- | :---- | :---- |
| **A (Baseline)** | Normal | X \\sim \\mathcal{N}(0, I) | None | Standard Double Descent curve (Peak at p \\approx n). |
| **B (Skewed)** | Log-Normal | X \\sim \\text{LogNormal}(0, 1\) | Heavy tails, outliers | Peak will be wider/higher due to difficulty interpolating outliers. |
| **C (High-Card)** | Categorical | 5 features, K=100 levels each | Extreme sparsity | One-Hot Encoding shifts peak to left; Embeddings flatten the peak. |
| **D (Imbalanced)** | Normal | X \\sim \\mathcal{N}(0, I) | 90% Censoring | Peak shifts to p \\approx N\_{events} (Effective Sample Size). |

### **3.3 Python Implementation Strategy**

The data generation will be implemented in Python using numpy and scipy. A crucial component for Scenario C (Categorical) is the encoding step. We will implement a custom preprocessor to toggle between One-Hot and Embedding representations.  
`# Conceptual Implementation for Inverse Transform Sampling with Weibull`  
`import numpy as np`

`def generate_survival_data(n_samples, n_features, censoring_rate, dist_type='normal'):`  
    `# 1. Generate Covariates`  
    `if dist_type == 'normal':`  
        `X = np.random.normal(0, 1, (n_samples, n_features))`  
    `elif dist_type == 'lognormal':`  
        `X = np.random.lognormal(0, 0.5, (n_samples, n_features))`  
      
    `# 2. Define Linear Predictor (Ground Truth)`  
    `beta = np.random.uniform(-1, 1, n_features)`  
    `eta = np.dot(X, beta)`  
      
    `# 3. Inverse Transform Sampling (Weibull)`  
    `lambda_k, nu_k = 0.5, 2.0 # Scale and Shape`  
    `u = np.random.uniform(0, 1, n_samples)`  
    `T = (-np.log(u) / (lambda_k * np.exp(eta)))**(1/nu_k)`  
      
    `# 4. Censoring`  
    `# Calibrate lambda_c to match desired rate`  
    `# (Iterative search or analytical approx would be used here)`  
    `lambda_c = find_censoring_lambda(T, censoring_rate)`   
    `C = np.random.exponential(1/lambda_c, n_samples)`  
      
    `Y = np.minimum(T, C)`  
    `delta = (T <= C).astype(int)`  
      
    `return X, Y, delta`

## **4\. Experimental Design**

### **4.1 Model Architecture: Scalable DeepSurv**

The primary model for investigation will be **DeepSurv**, a feed-forward neural network that outputs a single node representing the log-risk \\hat{h}(x), trained to minimize the Cox partial likelihood. To observe double descent, we need a "control knob" for model capacity. We will use the **width (w)** of the hidden layers.

* **Architecture:** Input(d) \\to Dense(w) \\to ReLU \\to Dense(w) \\to ReLU \\to Linear Output(1).  
* **Parameter Scaling:** We will vary w such that the total number of parameters P sweeps from P \\ll n (under-parameterized) to P \\gg n (over-parameterized). specifically, we will test ratios P/n \\in \[0.1, 100\].

### **4.2 Training Protocol**

Double descent is sensitive to the training dynamics.

* **Optimization:** We will use **Adam** optimizer.  
* **Epochs:** Crucially, we will train for a **large number of epochs (e.g., 50,000)** without early stopping. Double descent often requires the model to be trained "to completion" (zero training loss) to manifest the second descent. Early stopping acts as implicit regularization and cuts off the curve before the interpolation threshold.  
* **Regularization:** We will compare two settings:  
  1. **No Regularization:** (No weight decay, no dropout). This is necessary to observe the "pure" phenomenon.  
  2. **Standard Regularization:** (Weight decay \\lambda \= 1e-4). Theory suggests this should dampen the peak.

### **4.3 Evaluation Metrics: The Divergence Hypothesis**

Standard survival analysis relies heavily on the **Concordance Index (C-index)**. However, we hypothesize that C-index is blind to the variance spikes characteristic of double descent because it is rank-based. We will employ a multi-metric evaluation strategy:

1. **Concordance Index (C-index):** Measures discrimination. \\text{C} \= P(\\hat{h}\_i \> \\hat{h}\_j | T\_i \< T\_j). Insensitive to calibration.  
2. **Integrated Brier Score (IBS):** Measures the mean squared error between the predicted survival probability S(t|x) and the observed status over time. The Brier Score penalizes overconfident predictions (poor calibration), which are typical at the interpolation threshold. We hypothesize that IBS will show the double descent peak clearly, while C-index may not.  
3. **Negative Log Partial Likelihood (Loss):** The direct optimization objective. This should theoretically track the double descent curve most closely.

## **5\. Anticipated Results and Analysis**

This section details the expected outcomes based on the intersection of deep learning theory and survival analysis mechanics.

### **5.1 The "Ghost" Peak in C-index (Metric Divergence)**

We anticipate a significant divergence between the C-index and the IBS.

* **Hypothesis:** At the interpolation threshold (p \\approx n), the model will "wiggle" aggressively to fit noisy survival times. This results in predicted risk scores \\hat{h}(x) with extremely large magnitudes (e.g., \\pm 100).  
* **Impact on C-index:** Since the C-index only cares about the *order* of the scores, if the "wiggles" preserve the relative ranking of high-risk vs. low-risk patients, the C-index may remain high or show only a minor dip.  
* **Impact on IBS:** The Brier Score depends on the predicted probability S(t|x) \= S\_0(t)^{\\exp(\\hat{h})}. Extreme \\hat{h} values will drive survival probabilities to exactly 0 or 1\. If these extreme predictions are driven by noise (overfitting), the Brier Score will explode, revealing the interpolation peak.  
* **Implication:** This result would demonstrate that C-index is a dangerous metric for selecting deep learning models, as it can select overfitted, uncalibrated models that are statistically unstable.

### **5.2 The Skewness Amplifier**

For Scenario B (Log-normal covariates), we expect the double descent peak to be **wider and higher** than in the Gaussian baseline.

* **Reasoning:** Log-normal distributions generate "leverage points"—samples with extreme covariate values. In the critical regime, the model must expend significant capacity (large weights) to interpolate these outliers. This increases the variance of the estimator across the entire input space, exacerbating the test error spike.  
* **Guidance:** This would empirically confirm the necessity of log-transforming skewed clinical variables, not just for convergence speed, but to stabilize the generalization landscape.

### **5.3 High Cardinality and the Embedding Shield**

For Scenario C (Categorical), we expect distinct behaviors based on encoding:

* **One-Hot Encoding:** The dimensionality d increases by \\sum K\_i. The model enters the over-parameterized regime very early (small width). The peak may occur almost immediately or be skipped entirely if the initialization puts the model in the "kernel regime." However, generalization will likely be poor due to sparsity.  
* **Entity Embeddings:** By projecting categories into dense vectors (e.g., dimension 5 or 10), embeddings reduce the effective parameter count. We anticipate that Embeddings will shift the interpolation threshold to the right (larger width required to overfit) and, crucially, **flatten the peak**. The continuous nature of the embedding space allows for smoother interpolation than the discrete one-hot space.

### **5.4 The Censoring Shift**

For Scenario D (Imbalance), we predict the most novel theoretical finding.

* **Hypothesis:** The interpolation threshold will occur near p \\approx N\_{events} rather than p \\approx N\_{total}.  
* **Reasoning:** Censored data points provide inequality constraints (T\_i \> C\_i), which are weaker than equality constraints (T\_i \= t\_i). Therefore, the "degrees of freedom" required to memorize the dataset are lower.  
* **Result:** In a dataset with 90% censoring (N=1000, E=100), we expect the error peak to occur when the model has approximately 100 effective parameters, not 1000\. This implies that "safe" low-capacity models might actually be squarely in the critical danger zone for rare disease datasets.

## **6\. Discussion and Clinical Implications**

### **6.1 Re-evaluating "Benign Overfitting" in Medicine**

The concept of "benign overfitting"—where massive models fit noise but generalize well—is a cornerstone of modern AI. However, our proposed study aims to determine if this holds for survival data. If our results show that the "second descent" in the Brier Score is shallow (i.e., the error doesn't drop significantly below the classical minimum), it implies that **overfitting is not benign in survival analysis**. This distinction is vital. In image classification, a 99.9% confidence prediction on a noisy image might be acceptable if the class is correct. In survival analysis, predicting a 99.9% probability of death at 1 year for a patient who actually has a 50% chance is a catastrophic calibration failure that could lead to palliative care being wrongly recommended. If deep survival models are prone to this uncalibrated confidence in the over-parameterized regime, they require strict post-hoc calibration (e.g., Platt scaling) before deployment.

### **6.2 Guidelines for Model Selection**

Based on the anticipated findings, this manuscript will offer concrete guidelines for practitioners:

1. **Metric Integrity:** Never rely solely on C-index for deep survival models. The Brier Score must be the primary metric for assessing generalization stability.  
2. **Capacity Planning:** When working with heavy censoring, estimate model capacity limits based on the number of *events*, not the total cohort size.  
3. **Data Preprocessing:** High-cardinality variables should be handled via embeddings or target encoding to avoid rapid entry into the high-variance critical regime associated with sparse one-hot encoding.

### **6.3 Limitations and Future Work**

This study focuses on the Cox PH framework. Future work should extend this stress-testing to discrete-time models (e.g., DeepHit) and fully parametric models (e.g., Deep AFT), which may have different interpolation characteristics. Additionally, we assume non-informative censoring; investigating double descent under informative censoring (where censoring predicts risk) would be a valuable extension for real-world EHR data analysis.

## **7\. Conclusion**

The "Double Descent" phenomenon represents a frontier in our understanding of how deep learning models generalize. While well-documented in other fields, its implications for Time-to-Event analysis—with its unique loss functions and data structures—remain unmapped. This proposal outlines a rigorous, simulation-based methodology to chart this territory. By systematically varying data distribution types (Normal, Skewed, Categorical) and censoring rates, we aim to uncover the specific conditions under which deep survival models succeed or fail.  
The expected findings—specifically the divergence between rank-based and calibration-based metrics and the shift of the interpolation threshold due to censoring—promise to refine the theoretical foundations of computational medicine. Ultimately, this work seeks to transform the "art" of tuning deep survival models into a "science," providing the safety guardrails necessary for the responsible deployment of AI in patient care.

## **Appendix A: Mathematical Derivation of Synthetic Generation**

### **A.1 Inverse Transform Sampling for Weibull-Cox**

We assume the hazard function:  
The cumulative hazard is:  
The survival function is:  
To generate a random survival time T, we draw U \\sim \\text{Uniform}(0, 1\) and solve S(T) \= U:  
This derivation confirms that T follows a Weibull distribution with scale parameter depending on \\mathbf{x}. By controlling \\nu (shape), we can simulate increasing (\\nu \> 1\) or decreasing (\\nu \< 1\) hazards.

### **A.2 Simulating Correlated Categorical Data**

To generate a categorical variable X\_{cat} with K levels correlated with a continuous variable X\_{cont}, we use a Gaussian Copula.

1. Generate \\begin{bmatrix} Z\_1 \\\\ Z\_2 \\end{bmatrix} \\sim \\mathcal{N}\\left( \\begin{bmatrix} 0 \\\\ 0 \\end{bmatrix}, \\begin{bmatrix} 1 & \\rho \\\\ \\rho & 1 \\end{bmatrix} \\right).  
2. Let X\_{cont} \= Z\_1 (Standard Normal).  
3. Let U\_2 \= \\Phi(Z\_2) (Uniform).  
4. Define cutoffs q\_0, q\_1, \\dots, q\_K based on the desired marginal probabilities of the categories (e.g., uniform marginals imply q\_k \= k/K).  
5. Set X\_{cat} \= k if q\_{k-1} \\le U\_2 \< q\_k. This ensures X\_{cat} has the correct marginal distribution and a correlation of approximately \\rho with X\_{cont} in the latent space.

**Table 2: Proposed Experimental Grid**

| Experiment | Dataset | Varying Parameter | Model | Metrics Recorded |
| :---- | :---- | :---- | :---- | :---- |
| **Exp 1: Baseline** | Gaussian | Width (k=2 \\dots 2048\) | DeepSurv (Unreg) | C-index, IBS, NLL, Grad Norm |
| **Exp 2: Skew** | Log-Normal (\\sigma=0.5, 1.0, 1.5) | Width (k) | DeepSurv (Unreg) | C-index, IBS, NLL |
| **Exp 3: Sparsity** | Categorical (K=100, 1000\) | Width (k) | DeepSurv (Embedding vs OneHot) | C-index, IBS, Time-to-Peak |
| **Exp 4: Imbalance** | Gaussian | Censoring Rate (c=20\\%, 50\\%, 90\\%) | DeepSurv (L2 Reg vs Unreg) | C-index, IBS, Peak Location |

#### **Works cited**

1\. Understanding Overparametrization in Survival Models through Double-Descent \- arXiv, https://arxiv.org/html/2512.12463v1 2\. Deep double descent | OpenAI, https://openai.com/index/deep-double-descent/ 3\. Double descent \- Wikipedia, https://en.wikipedia.org/wiki/Double\_descent 4\. Deep Double Descent: Where Bigger Models and More Data Hurt \- OpenReview, https://openreview.net/forum?id=B1g5sA4twr 5\. Characterizations of Double Descent \- SIAM.org, https://www.siam.org/publications/siam-news/articles/characterizations-of-double-descent/ 6\. Exact expressions for double descent and implicit regularization via surrogate random design \- UC Berkeley Statistics, https://www.stat.berkeley.edu/\~mmahoney/pubs/NeurIPS-2020-double-descent.pdf 7\. \[2512.12463\] Understanding Overparametrization in Survival Models through Double-Descent \- arXiv, https://arxiv.org/abs/2512.12463 8\. Understanding Overparametrization in Survival Models through Double-Descent, https://www.researchgate.net/publication/398719874\_Understanding\_Overparametrization\_in\_Survival\_Models\_through\_Double-Descent 9\. Survival Analysis and Interpretation of Time-to-Event Data: The Tortoise and the Hare \- PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC6110618/ 10\. Survival modeling using deep learning, machine learning and statistical methods: A comparative analysis for predicting mortality after hospital admission \- arXiv, https://arxiv.org/pdf/2403.06999 11\. High-dimensional, massive sample-size Cox proportional hazards regression for survival analysis \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC3944969/ 12\. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC5828433/ 13\. Generating Accurate Synthetic Survival Data by Conditioning on Outcomes \- arXiv, https://arxiv.org/html/2405.17333v2 14\. Deep Neural Networks for Survival Analysis Using Pseudo Values \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC8056290/ 15\. Log-Normal Distribution with Python | by Gianpiero Andrenacci | AI Bistrot | Medium, https://medium.com/data-bistrot/log-normal-distribution-with-python-7b8e384e939e 16\. Machine Learning with High-Cardinality Categorical Features in Actuarial Applications, https://ideas.repec.org/a/cup/astinb/v54y2024i2p213-238\_1.html 17\. A Guide to Handling High Cardinality in Categorical Variables | by Niranjan Appaji \- Medium, https://niranjanappaji.medium.com/a-guide-to-handling-high-cardinality-in-categorical-variables-7b4101d3af68 18\. Deep learning for survival outcomes \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC7334068/ 19\. DeepSurv is a deep learning approach to survival analysis. \- GitHub, https://github.com/jaredleekatzman/DeepSurv 20\. Understanding Overparametrization in Survival Models through Interpolation \- arXiv, https://arxiv.org/html/2512.12463v2 21\. Sample size and predictive performance of machine learning methods with survival data: A simulation study \- AIR Unimi, https://air.unimi.it/retrieve/83306d47-5fcd-4ad8-9977-00cd3c11d4e0/Statistics%20in%20Medicine%20-%202023%20-%20Infante%20-%20Sample%20size%20and%20predictive%20performance%20of%20machine%20learning%20methods%20with%20survival.pdf 22\. Generating survival times to simulate Cox proportional hazards models \- PubMed, https://pubmed.ncbi.nlm.nih.gov/15724232/ 23\. Large no of categorical variables with large no of categories \- Data Science Stack Exchange, https://datascience.stackexchange.com/questions/53181/large-no-of-categorical-variables-with-large-no-of-categories 24\. machine learning \- best way to create Synthetic data generation, https://datascience.stackexchange.com/questions/134428/best-way-to-create-synthetic-data-generation 25\. Utility-based Analysis of Statistical Approaches and Deep Learning Models for Synthetic Data Generation With Focus on Correlation Structures: Algorithm Development and Validation \- JMIR AI, https://ai.jmir.org/2025/1/e65729/ 26\. Simulating correlated lognormals in Python \- numpy \- Stack Overflow, https://stackoverflow.com/questions/58633930/simulating-correlated-lognormals-in-python 27\. Generate multivariate distributions of lognormal and normal distribution in python, https://stats.stackexchange.com/questions/632421/generate-multivariate-distributions-of-lognormal-and-normal-distribution-in-pyth 28\. How to simulate datasets from different survival models • survMS \- Mathilde Sautreuil, https://mathildesautreuil.github.io/survMS/articles/how-to-simulate-survival-models.html 29\. Simulating survival data to reflect existing complex dataset and censoring proportions, https://stats.stackexchange.com/questions/620307/simulating-survival-data-to-reflect-existing-complex-dataset-and-censoring-propo 30\. Censored Data Models — PyMC example gallery, https://www.pymc.io/projects/examples/en/latest/survival\_analysis/censored\_data.html 31\. Statistical Learning: 10.7 Interpolation and Double Descent \- YouTube, https://www.youtube.com/watch?v=qRHdQz\_P\_Lo 32\. Deep double descent: where bigger models and more data hurt \- Batista Lab, https://files.batistalab.com/teaching/attachments/chem584/Nakkiran\_2021\_J.\_Stat.\_Mech.\_2021\_124003.pdf 33\. Evaluating Survival Models — scikit-survival 0.26.0, https://scikit-survival.readthedocs.io/en/stable/user\_guide/evaluating-survival-models.html 34\. Pitfalls of the Concordance Index for Survival Outcomes \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC10219847/ 35\. SurvMetrics: An R package for Predictive Evaluation Metrics in Survival Analysis, https://journal.r-project.org/articles/RJ-2023-009/ 36\. A novel non-negative Bayesian stacking modeling method for Cancer survival prediction using high-dimensional omics data \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC11067084/ 37\. 4 ways to encode categorical features with high cardinality \- Towards Data Science, https://towardsdatascience.com/4-ways-to-encode-categorical-features-with-high-cardinality-1bc6d8fd7b13/ 38\. How to Evaluate Survival Analysis Models | by Nicolo Cosimo Albanese \- Medium, https://medium.com/data-science/how-to-evaluate-survival-analysis-models-dd67bc10caae 39\. \[2307.02071\] A Comparison of Machine Learning Methods for Data with High-Cardinality Categorical Variables \- arXiv, https://arxiv.org/abs/2307.02071 40\. Dynamic Survival Transformers for Causal Inference with Electronic Health Records | Request PDF \- ResearchGate, https://www.researchgate.net/publication/364814437\_Dynamic\_Survival\_Transformers\_for\_Causal\_Inference\_with\_Electronic\_Health\_Records 41\. Simulating time-to-event data subject to competing risks and clustering: A review and synthesis \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC11654122/