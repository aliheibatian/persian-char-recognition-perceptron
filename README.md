# Persian Character Recognition with Single-Layer Perceptron

Multi-class binary classification of handwritten Persian letters using 17 one-vs-rest perceptrons.

## Dataset
- 119 samples (7 per class × 17 classes)
- 95×95 pixels, binarized to bipolar (-1 / +1)
- Source image
<img width="326" height="802" alt="data" src="https://github.com/user-attachments/assets/9e842213-d273-4fa6-97b6-66327a2459d0" />


## Method
- One-vs-Rest strategy (17 separate perceptrons)
- Input: flattened 95×95 = 9025 features
- Learning rate: 0.01
- Convergence: stops when no misclassification in an epoch

## Results
- Training accuracy: 100%
- Convergence epochs per class visualized
<img width="800" height="443" alt="test1" src="https://github.com/user-attachments/assets/e7df49c8-3b88-46dc-8042-24c147d47770" />
<img width="796" height="402" alt="test2" src="https://github.com/user-attachments/assets/afd6f7b5-a07e-4be5-88ac-fae55b72da61" />
