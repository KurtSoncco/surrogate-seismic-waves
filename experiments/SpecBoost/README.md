This experiment aims to determine if a two-stage training methodology can improve the predictive accuracy of Fourier Neural Operators (FNOs). Drawing inspiration from the spectral analysis in "Toward a Better Understanding of Fourier Neural Operators: Analysis and Improvement from a Spectral Perspective", we hypothesize that separating the learning process into distinct stages can more effectively capture different spectral components of the data.

Reference: https://arxiv.org/abs/2404.07200

The methodology consists of two sequential training phases:

Stage 1 (Base Mapping): An initial FNO model is trained to learn the fundamental mapping from the input domain to the output domain.

Stage 2 (Refinement): The output from the first stage is then used as input for a second FNO, which is trained to learn and correct the residual error.

The central goal is to observe if this hierarchical approach allows the model to better mitigate issues like spectral bias and achieve a more robust approximation of the target function.

Ideas:
- Since we have a mismatch between input shapes for the second stage of the Specboost model, we are going to use fusion branches to join both models. The same idea as we did before: Using the latent representation of the Vs, adding the same latent dimension extraction for the residuals, and combine them for the model G_b.


Conclusion:
Even after implementing several stages of boosting models, the residuals after the first Model A are hard and uncorrelated for a model to learn. After different debugging methods, the conclusion remained the same. The order of magnitude of residual of prediction is 1 lower of the required one. 