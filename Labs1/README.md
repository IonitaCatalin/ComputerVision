# Requirements

**1bp** 1) Using numpy indexing implement a generic 2D convolution operation working on single channel inputs. Consider using both indexing and block matrix operations to compute the convolution output. Compare the runtime against the dimensions of the convolution filter, keeping the same image dimensions. Plot the result of the convolution on each channel using subfigures from matplotlib.

**1bp** 2) Using the generic convolution operation, implement different image gradients and different smoothing filters. Plot the filtered output against the image input. Why are the results becoming more/less sharp, after each filtering operation?

**2bp** 3) Given a noisy image, apply Otsu binarization on the noisy image and a gaussian filtered output of the same image. Discuss the results given the particularities of Otsu's algorithm. Discuss about the two phases of the iterative procedure, and how the procedure converges to the given solution.
