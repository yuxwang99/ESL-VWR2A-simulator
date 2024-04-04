# FIR-Filter 2-Column Kernel

## Kernel description

This kernel uses both kernels of VWR2A to run a Finite Impulse Response (FIR) filter on the input data.

## Kernal usage

Inputs:
* A vector of data to be filtered. Its size should be divisible by twice VWR size of the VWR2A (i.e. 2x128 = 256).
* The "taps" (i.e. filter coefficients) to convolve with the input

Outputs:
* The filtered vector (same size as the input vector)

## Implementation details

First, the filter coefficients of the CGRA are written to an SPM bank, whose index is stored in SRF position 0 of both columns.The index of the data (both for reading and writing) in the SPM is stored to SFR position 1. The number of filter coefficients is stored into SRF position 3. The number of iterations of filtering, which is equal to INPUT_SIZE/CGRA_VWR_SIZE/2, is stored in SRF index 4. The output is stored to the same data index as the input.

## Examples of applications using this kernel

* [FIR filter example](https://eslgit.epfl.ch/esl/architectures-and-systems/accelerators/cgra/vwr2a_kernel_examples/-/tree/main/fir_filter/src/fir_filter.c)

