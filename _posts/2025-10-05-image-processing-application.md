---
title: "Image Processing Application (MATLAB)."
date: 2025-10-05
categories: [Projects,Image Processing]
tags: [Projects,Image Processing,MATLAB,from-scratch]
---

# Image Processing Application

This MATLAB-based image processing application provides a robust platform for real-time image manipulation and analysis. Built entirely from scratch without relying on MATLAB's built-in image processing functions, it features a custom graphical user interface (GUI) and a comprehensive set of algorithms for image enhancement, filtering, and feature extraction.

## Features

- **Custom GUI**: Intuitive interface for loading, processing, and saving images, with side-by-side display of original and processed images alongside their histograms.
- **Point Operations**: Includes contrast adjustment, histogram equalization, additive offset, multiplicative scaling, image inversion, and thresholding for precise pixel-level transformations.
- **Noise Addition**: Implements Gaussian and Salt-and-Pepper noise for testing image robustness.
- **Frequency-Domain Filters**: Supports ideal and Butterworth low-pass/high-pass filters, band-pass, band-reject, homomorphic, and local spectral filters for advanced frequency-based processing.
- **Spatial Filters**: Offers linear (3x3 and 5x5 mean, Gaussian, pyramidal, conical) and non-linear (median) filters for smoothing and noise reduction.
- **Edge Detection**: Includes Gradient, Sobel, Prewitt, Robert, Laplacian, Canny, Kirsch, and Marr-Hildreth methods for detecting image edges and features.
- **Morphological Operations**: Provides erosion, dilatation, opening, closing, and internal/external/morphological gradients for shape analysis and segmentation.
- **Interest Point Detection**: Implements SUSAN and Harris detectors for identifying key image features.

## Getting Started

### Prerequisites

- MATLAB
- No additional toolboxes required, as all functions are custom-built

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ammarlouah/image-processing-app.git
   ```
2. Open MATLAB and navigate to the project directory.
3. Run the main script:

   ```matlab
   ImageProcessingApp
   ```

### Usage

1. Launch the application by running the `.m` file in MATLAB.
2. Use the **File** menu to open an image (supports `.jpg`, `.png`, etc.).
3. Select operations from the menus (e.g., Point Operations, Noise, Filters, Morphology) to process the image.
4. View the original and processed images with their histograms in the GUI.
5. Save the processed image via the **File &gt; Save** option.

## Screenshot

![Application Screenshot](/assets/img/posts/image-processing-application/application.png)

## Project Structure

- `ImageProcessingApp.m`: Main script containing the GUI and all image processing functions.
- No external dependencies or libraries used—everything is implemented from scratch.

## Notes

- The application is designed for grayscale and RGB images, with automatic handling of image formats.
- All algorithms are optimized for performance while maintaining accuracy.
- The code is fully commented for clarity and maintainability.

## License

This project is licensed under the MIT License.

## Contact

For questions or contributions, reach out via GitHub Issues or email me at ammarlouah9@gmail.com
