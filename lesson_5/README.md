## 01. MatrixMultiplication (OpenMP)

### Performance reference table (N = 1024):

CPU time   | GPU time | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
57194 ms   | 25 ms    | 2198x     | Nvidia Jetson TX2  | Shared Mem | -
52100 ms   | 84 ms    | 614x     | Nvidia Jetson TX2  | No Shared | -
8769 ms   |  -    | 3.5x   | Nvidia Jetson TX2  | OpenMP (-O3) | -
10224 ms  |  -    | 3.6x   | Nvidia Jetson TX2  | OpenMP (-O0) | -

## 02. Factorial

### Performance reference table (N = 268435456):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
7546 ms    | 1725 ms  | 4.3x     | Nvidia Jetson TX2  | OpenMP | -


## 03. Find

Find two (given) consecutive numbers in an array.

### Performance reference table (N = 67108864):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
120 ms     | 53 ms    | 2.2x     | Nvidia Jetson TX2  | OpenMP | -

## 04. RC4 Chiper

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l5_04.jpg" width="500" height=auto> 

### Performance reference table (N = 256):

CPU time   | CPU time* | Speedup  | Device             | Mode  |Author
-----------| -------- | -------- | ------------------ | ----  |------
29705 ms   | 26834 ms    | x     |   | OpenMP | -
