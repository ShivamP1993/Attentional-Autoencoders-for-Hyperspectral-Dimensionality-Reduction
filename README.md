# Attentional Autoencoders for Hyperspectral Dimensionality Reduction

Hyperspectral images (HSIs) are being actively used for landuse/land cover classification owing to their high spectral resolution. However, this leads to the problem of high dimensionality, making the algorithms data hungry. To resolve these issues, deep learning techniques, such as convolution neural networks (CNNs) based autoencoders, are used. However, traditional CNNs tend to focus on all the features irrespective of their importance, leading to weaker representations. To overcome this, we incorporate attention modules in our autoencoder architecture. These attention modules explicitly focus on more important wavelengths, leading to better transformation of the features in the low dimension. In the proposed method, the attention driven encoder transforms high dimension features to low dimensions, considering their relative importance, while the CNN based decoder reconstructs the original features. We evaluate our method on Indian pines 2010 and Indian pines 1992 hyperspectral datasets, where it surpasses the previous approaches.