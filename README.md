# Deformable Attention Operation Optimization

## Objective
The goal of this project is to showcase the optimization of the Deformable Attention Operation on a custom chip architecture using AVX2 instructions. The project aims to demonstrate the ability to identify performance bottlenecks and implement efficient solutions for matrix multiplication and deformable attention operations.

## Technologies Used
- C++: The programming language used for implementing the matrix multiplication and deformable attention operations.
- Eigen Library: Eigen is a C++ template library for linear algebra that provides efficient implementations of various matrix operations, including AVX2-optimized matrix multiplication.
- AVX2 Instructions: Advanced Vector Extensions 2 (AVX2) is an extension to the x86 instruction set that enables performing parallel operations on multiple data elements simultaneously, which can significantly improve the performance of certain computations.

## Project Structure
The project is organized into the following files:

1. `main.cpp`: The main entry point of the application. It includes benchmarking code for scalar matrix multiplication, AVX2-optimized matrix multiplication, and deformable attention operation.
2. `matrix_multiplication.h` and `matrix_multiplication.cpp`: These files contain functions for scalar and AVX2-optimized matrix multiplication using Eigen.
3. `deformable_attention.h` and `deformable_attention.cpp`: These files contain the implementation of the deformable attention operation.

## How to Run
To compile and run the project, follow these steps:

1. Make sure you have a C++ compiler installed on your system (e.g., GCC).

2. Download or clone this repository to your local machine.

3. Open a terminal (e.g., Git Bash) and navigate to the project directory.

4. Compile the code using the following command:

   ```bash
   g++ -O0 -mavx2 main.cpp matrix_multiplication.cpp deformable_attention.cpp -o matrix_benchmark
   ```
   The -O0 flag disables compiler optimizations, and the -mavx2 flag enables AVX2 instructions.

5. Run the executable:
    ```bash
    ./matrix_benchmark
    ```

6. The program will display the durations for scalar matrix multiplication, AVX2-optimized matrix multiplication, and the deformable attention operation in microseconds.

## Results

When you run the matrix_benchmark program, you should observe the durations for each operation. The AVX2-optimized matrix multiplication and deformable attention operation should have significantly lower durations compared to the scalar matrix multiplication. This demonstrates the successful optimization of the operations using AVX2 instructions.

## Notes

- The matrix size used for benchmarking can be modified in the main.cpp file by changing the value of N.
- For better performance comparison, it's recommended to run the benchmark on a machine that supports AVX2 instructions.

## Credits 

This project is inspired by the paper: [Deformable Attention Operation](https://arxiv.org/pdf/2201.00520.pdf)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.