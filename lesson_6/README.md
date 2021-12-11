## 00. Fibonacci

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_00.jpg" width="500" height=auto> 

### Performance reference table (N = 44):

CPU time   | CPU* time | Speedup  | Device             | Mode         | Author
-----------| --------  | -------- | ------------------ | -----------  |------
6039 ms    | 3758 ms   | 1.6x     | Intel i5-7400      | OpenMP (-O0) | Filippo Nevi

## 01. QuickSort

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_01.jpg" width="500" height=auto> 

### Performance reference table (N = 1 << 20):

CPU time   | CPU* time | Speedup  | Device             | Mode         | Author
-----------| --------- | -------- | ------------------ | -----------  |------
12558 ms   | 12736 ms  | 0.98x	  | Intel i5-7400      | OpenMP (-O0) | Filippo Nevi

## 02. Producer Consumer

<img src="https://github.com/PARCO-LAB/Advanced-Computer-Architectures/blob/main/figures/l6_02.jpg" width="500" height=auto> 

Tested on a Raspberry Pi 3 A+ with a 1.4GHz quad-core CPU.

1) Critical Regions: avg speedup = 1.6x
2) Locks: avg speedup = 1.84x

