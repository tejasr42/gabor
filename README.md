# gabor
Python implementation of the Gabor-jet model

This uses the Fast-Fourier Transform to sample the Gabor kernel (8 orientations, 5 scales) across 100 locations in the images. Euler distance between Gabor vectors is then used to calculate image similarity.

Next steps: See if the filters can be learned using a convolutional network
