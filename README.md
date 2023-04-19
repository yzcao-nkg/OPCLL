# Consistent Complementary-Label Learning via Order-Preserving Losses

## Requirements
- Python 3.6
- numpy 1.14
- PyTorch 1.1
- torchvision 0.2

## Demo
The following demo will show the results of Order-preserving Complementary-Label Learning with the Kuzushiji-MNIST dataset. When running the code, the test accuracy of each epoch will be printed for `OP-W` and `OP`. The results will have two columns: epoch number and test accuracy.


Before running `demo.py`, we can choose the type of method. If we run the following code:

```bash
python demo.py -me OP 
```
the method OP will be used or OP-W will be used in default.
