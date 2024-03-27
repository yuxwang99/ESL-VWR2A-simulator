# Morphological Filter Baseline Removal Kernel

## Kernel description

This kernel performs the Cooley-Turkey algorithm to compute a Fast Fourier Transform (FFT) of a vector of data. The BITREV_SPLITOPS kernel can be called directly afterwards to compute the final FFT result. Alternatively, the BITREV_SPLITOPS_MAGNITUDE_SQUARED kernel can be called to comptue the periodogram.

## Kernal usage

### In the case of a complex-valued FFT:
Inputs:


Outputs:


## Implementation details

Whoever figures out how this kernel works, please update the README here :)

## Examples of applications using this kernel

* [Morphological Filter Baseline Removal](https://eslgit.epfl.ch/esl/architectures-and-systems/accelerators/cgra/vwr2a_kernel_examples/mf_baseline_rm/src/morph_filter.c)
* [Queue Baseline Removal](https://eslgit.epfl.ch/esl/architectures-and-systems/accelerators/cgra/vwr2a_kernel_examples/queue_baseline_rm/src/morph_filter.c)
* [Morphological Baseline Lowpass Filter](https://eslgit.epfl.ch/esl/architectures-and-systems/accelerators/cgra/vwr2a_kernel_examples/mf_baseline_lp_filter_cgra_1l/src/morph_filter.c)



