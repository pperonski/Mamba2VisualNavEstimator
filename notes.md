- Fast KAN is faster than Efficient KAN so lets stick with it.
- Convolution KAN is a lot slower than standard convolution
- So we need to think about diffrent approach ( we can replace activation function with KAN or just replace convolution with KAN )
- In the end activation for Conv1D layer has been replaced by noraml KAN layer.
- I am going with 3D map estimation since it has more sense
- Use open3d library for processing point map
- Lets stick with size of 10x10x10 meters output cube
- I also had idea for making a network just split points with coordinates to put into map but I am not so sure
about it, for large map like colosseum it would yield too much points
- we could add additional KAN network at the output that will project Mamba KAN
to a finite large set of points, I think I will got with that, what possible can go wrong?