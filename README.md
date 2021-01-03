# Small-NN-Learns-MNIST-Dataset
MNIST is a Dataset that contains 70000 (60000 Train and 10000 Test) Labeled Images with Handwritten Digits.  
I made a small neural network with my [Simple Neural Network Library](https://github.com/irineos/simple-Neural-Network-library-in-C) and trained it in this Dataset. 

## Run  
To train the model  

    gcc trainMNIST.c -o trainMNIST -lm  && ./trainMNIST  
  
Test the Dataset with your saved model or with the pre trained model ("mnist_model.txt")  

    gcc testMNIST.c -o test -lGL -lGLU -lglut -lm && ./test  
    
   ![alt test]()
    
  
You can also draw your own digits to test further your model's capabilities  

    gcc paintDigit.c -o paint -lGL -lGLU -lglut -lm && ./paint
  
   ![alt test]()
