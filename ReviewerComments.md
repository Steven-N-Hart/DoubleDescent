Dear Dr Hart,

Your manuscript, "The Survival Double Descent: Generalization Dynamics of Deep Neural Networks in Time-to-Event Analysis", has now been assessed.

*****
Handling Editor: Niansheng Tang
*****

We invite you to revise your paper, carefully addressing the comments from the reviewers and the editor. Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained. When your revision is ready, please submit the updated manuscript and a point-by-point response. This will help us move to a swift decision.

---

Reviewer 1

The overall paper is well-written and effectively illustrates how machine learning can address survival analysis problems. It also provides a systematic simulation design that connects the results to existing literature. Specifically, their findings suggest that deep learning-based Cox survival model selection, relying solely on discrimination metrics such as the C-index, may overlook calibration issues.
Despite these overall strengths, minor revisions are needed before accepting it for publication in BMC Medical Research Methodology, as outlined below.
1. The simulation used appears to represent a relatively simple survival problem with few
predictive covariates, even though the goal was to explore the double descent phenomenon in a deep survival model. Based on this setup, the underlying hazard behavior can be easily modeled by a very small network (2 neurons wide) or even a simple linear Cox model, as shown. Therefore, it is important to include clarifications on how the problem's complexity relates to the required model capacity.
2. Clarification is needed on whether the interpolation threshold observed in the simulation reflects the true dynamics of survival learning or is merely an artifact of overparameterization applied to a simple problem.
3. Additional experiments with deep survival architectures could be conducted to verify whether the claim that neural-survival deep learning architectures exhibit double descent is accurate.
4. The comment about the sparsity of categorical covariates leading to performance close to
chance might not be sufficient. I recommend that the authors review their simulation setup for these covariates, as applying quantile binning to continuous variables from a Gaussian copula could unintentionally discard important predictor information.
5. If possible, the authors should demonstrate their results on real-world data, as survival data are generally more complex than controlled simulation settings.

Attachment(s):

Download Reviewer 1 attachment 1
Reviewer 2

This paper addresses the double descent problem, which is a practical issue in ML. They empirically show that this problem can be also shown in survival set up by providing various simulations.

Major Comments
1. On page 2, the functions f and f* are introduced but not clearly defined. Please provide a more precise definition of both functions.
2. On pages 11 and 12, the references supporting the theoretical predictions appear to be missing. Please include the appropriate citations.
3. The authors should provide more explanation as to why the second descent is attenuated compared with standard supervised classification settings.
4. Please add a brief introduction to post-hoc recalibration and direct probability prediction methods for readers who may not be familiar with these approaches.
5. Scenario C appears to be too strong relative to the other scenarios, which may make the comparison less informative. The authors may consider simulating more realistic situations, such as datasets with mixed covariates (e.g., some continuous variables and some categorical variables).

Major comments:
1. Tables should be properly numbered throughout the manuscript.
2. On page 12, please clarify whether P represents the number of predictors.
3. On page 13, the citation “Harrell ()” is incomplete. Please provide the correct year (likely Harrell, YYYY) and ensure the reference is properly formatted.

Reviewer 3

Overall comments:
This paper investigates the double descent problem under the Cox PH model with Cox’s partial likelihood. For this purpose, the authors present the simulation study based on synthetic survival data with several scenarios. This is an interesting topic, but further justifications are required.

Main comments:
1) Please clearly present the problems caused by double descent in survival analysis, together with mathematical procedures. I am not sure why it is important to check the double descent. When training deep neural networks (DNNs) such as DeepSurv, it is essential to tune hyperparameters, including regularization. For example, the loss function of DeepSurv includes the regularization, i.e. L2 penalty.

2) It is well known that the C-index, when evaluating models such as DeepSurv trained with the Cox partial likelihood, is insensitive to extreme risk scores, whereas the Integrated Brier Score (IBS) is sensitive to them. You should cite the related references.

3) P5, Data Generation: The survival a are generated from the Cox-PH model where the log-risk score is linear. Why do you use the linear assumption? DeepSurv typically considers non-linear assumption in the framework of the DNNs. Random survival forest (RSF) also handles the non-linear case.

4) P5, Data Generation: The authors state that “Real datasets lack the controlled conditions required to trace the double descent curve. We therefore generate synthetic survival data.”
I do not agree with the statement. It is very important to present the double descent problem in real-world data.
