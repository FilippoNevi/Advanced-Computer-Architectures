## 01. MatrixMultiplication (OpenMP)

### Performance reference table (N = 1024):

CPU time   | CPU time* | Speedup  | Device         | Mode   |Author
-----------| --------  | -------- | -------------- | ----   |------
3555 ms    |  927 ms   | 3.8x     | Intel i5-7400  | OpenMP | Filippo Nevi

## 02. Factorial

### Performance reference table (N = 268435456):

CPU time   | CPU time* | Speedup  | Device             | Mode   |Author
-----------| --------  | -------- | ------------------ | ----   |------
7546 ms    | 1725 ms   | 4.3x     | Nvidia Jetson TX2  | OpenMP | Filippo Nevi 


## 03. Find

Find two (given) consecutive numbers in an array.

### Performance reference table (N = 67108864):

CPU time   | CPU time* | Speedup  | Device             | Mode   |Author
-----------| --------  | -------- | ------------------ | ----   |------
120 ms     | 53 ms     | 2.2x     | Nvidia Jetson TX2  | OpenMP | Filippo Nevi 

## 04. RC4 Cipher

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l5_04.jpg" width="500" height=auto>

### Performance reference table (N = 256):

CPU time   | CPU time* | Speedup  | Device             | Mode   |Author
-----------| --------  | -------- | ------------------ | ----   |------
1497 ms    | 3406 ms   | 0.44x    | Intel i5-7400      | OpenMP | Filippo Nevi

Note: for the parallel version of RC4, the program sometimes cannot crack the key, and when it cracks it, the execution time changes drastically between one instance and another. The highest time I've recorded in five attempts was 4424 ms, while the least time was 698 ms.
