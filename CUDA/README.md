## CUDA folder 
## Names: Anuradha Agarwal, Thoomas Keller, Zack Humphries 

This folder contains the code used to parallelize the deep neural network using CUDA. The following are the files present in the folder:
- main.c: main file that parallelized using CUDA
- test_data4801.csv: test data with 4800 inputs
- train_data4801.csv: train data with 4800 inputs 

# How to run
There is 1 command line argument: number of threads
- How to compile:
```
   	nvcc -o main main.cu
```
- How to run:

```
	rsh node10
	go to CUDA directory
   	./main <numberofThreads>
```
