# Pulsar Detection in Astronomy with the Murchison Widefield Array (MWA)

In the field of astronomy, it is common to detect signals amidst significant background noise. The following example addresses the question: How many pulsars are detected in images captured by the Murchison Widefield Array (MWA) telescope? The MWA is a low-frequency radio telescope situated in Western Australia. It detects radio emissions at frequencies ranging from 80 to 300 megahertz, comparable to those of popular radio stations. The telescope possesses a very large field of view, making it well-suited for extensive survey projects.

A typical image produced by the MWA appears as shown below.
![MWA Typical Image](Assets/Image1.png)


The grayscale intensity measures the flux density of emissions from astronomical objects, where black represents high flux density and gray indicates background noise. Most black dots observed in such images are distant radio galaxies. However, some correspond to objects within our own galaxy, such as pulsars or supernova remnants. In radio astronomy, flux density is quantified in units of Janskys, equivalent to $\(10^{-26}\)$ watts per square meter per hertz. In other words, flux density measures the spectral power received by a telescope detector per unit projected area. Additional details are available in the resources accompanying this video. For the purposes of this exercise, it is sufficient to understand that the objective is to measure the apparent brightness of a pulsar at a specific frequency.

One notable aspect is the faintness of astronomical objects compared to terrestrial sources like mobile phones. Astronomical images are typically stored in a file format known as FITS and can be viewed using software such as DS9 or online tools like Aladin. While some individuals prefer to display these images in false color, it is important to remember that radio frequencies do not inherently possess color. These color maps are employed solely to highlight different aspects of the intensity scale.

The current display features small cutouts from extensive radio images. The image on the left shows a detected pulsar at the center, whereas the image on the right shows no detection. A detection is generally defined as a flux density exceeding five standard deviations above the local noise level. Consequently, when searching for radio emissions at MWA frequencies at the locations of all known pulsars, detections are occasionally found, though they are more commonly absent.

When a pulsar is not detected, several factors may be responsible. The pulsar might be too distant, its intrinsic emission may be weak at these frequencies, or its emission could be intermittent and temporarily inactive. One might assume that no conclusions can be drawn from non-detections; however, astronomers have developed sophisticated techniques to extract meaningful information from such instances. One such technique, known as stacking, enables the measurement of the statistical properties of a population that cannot be individually detected.

Stacking is effective because the noise in a radio image is approximately random and follows a Gaussian distribution centered at zero. When regions of an image containing only noise are aggregated, the random variations tend to cancel out. In contrast, regions containing signals accumulate their emissions, thereby increasing the signal-to-noise ratio.

This concept can be illustrated using one-dimensional signals. Consider a signal resembling a single Gaussian curve with added random noise. With sufficient noise, the signal becomes indiscernible. However, if one averages 100 such signals, each with unique random noise, the signal-to-noise ratio improves, allowing the underlying signal to emerge across the entire population of 100 sources.

Applying this method to pulsar detection presents a challenge: undetected pulsars are dispersed across the sky. To effectively stack these signals, their positions must first be aligned by centering them on the same pixel. The stacking process involves calculating the mean stack by averaging every pixel in the image, thereby forming a new image from the result. This procedure is outlined in the subsequent set of activities.

During the project, the student advanced to this stage, where the process was relatively straightforward. However, a simple suggestion introduced complexity: it was recommended to perform a median stack instead of a mean stack, as the median is a more robust statistic. This refers to the fact that, as commonly learned in high school mathematics, the mean is more susceptible to outliers than the median. In symmetric distributions, both the mean and median yield identical results. However, in cases of asymmetric distributions or when significant outliers are present, the median provides a more accurate representation of the central value. The next lecture will explore how this seemingly minor methodological change significantly impacted the computational efficiency of obtaining results.


# Key Points: Detecting Pulsars with the Murchison Widefield Array (MWA)

## Introduction to Signal Detection in Astronomy
- **Challenge**: Detecting signals amidst noise.
- **Example Problem**: Determining the number of pulsars detected in MWA images.

## Murchison Widefield Array (MWA) Overview
- **Location**: Western Australia.
- **Frequency Range**: 80 to 300 MHz (similar to popular radio stations).
- **Features**:
  - Large field of view.
  - Ideal for large survey projects.

## Understanding MWA Images
- **Image Representation**:
  - Grayscale indicates flux density of astronomical objects.
  - **Black**: High flux density.
  - **Gray**: Background noise.
- **Common Objects**:
  - Distant radio galaxies (black dots).
  - Galactic objects like pulsars and supernova remnants.

## Flux Density in Radio Astronomy
- **Unit**: Janskys (Jy).
  - 1 Jy = 10⁻²⁶ W/m²/Hz.
- **Definition**: Spectral power received by a telescope detector per unit area.
- **Importance**: Measures the apparent brightness of pulsars at specific frequencies.

## Image Formats and Visualization
- **File Format**: FITS (Flexible Image Transport System).
- **Viewing Tools**:
  - Software: DS9.
  - Online Tools: Aladin.
- **False Color Display**:
  - Enhances intensity scale.
  - Radio frequencies inherently have no color.

## Pulsar Detection in MWA Images
- **Detection Criteria**:
  - Flux density > 5 standard deviations above local noise.
- **Detection Examples**:
  - Images with and without pulsar detections.
- **Non-Detections**:
  - Possible Reasons:
    - Pulsar too distant.
    - Weak intrinsic emission at MWA frequencies.
    - Intermittent or switched-off emission.

## Techniques for Analyzing Non-Detections
- **Stacking**:
  - Measures statistical properties of undetected populations.
  - **Process**:
    1. Align pulsar positions to the same pixel.
    2. Combine multiple images to enhance signal-to-noise ratio.
- **Signal vs. Noise**:
  - Noise is random with a Gaussian distribution centered on zero.
  - Stacking cancels out random noise, amplifies consistent signals.

## Stacking Demonstration
- **Single Signal with Noise**:
  - Signal becomes obscured with enough noise.
- **Multiple Signals (e.g., 100)**:
  - Averaging increases signal-to-noise ratio.
  - Underlying signal becomes detectable across the population.

## Challenges with Stacking
- **Position Alignment**:
  - Undetected pulsars are scattered across the sky.
  - Must shift positions to center on the same pixel before stacking.

## Mean Stack vs. Median Stack
- **Mean Stack**:
  - Calculates the average of each pixel across images.
  - Susceptible to outliers.
- **Median Stack**:
  - Uses the median value of each pixel.
  - More robust against outliers and asymmetric distributions.
- **Impact**:
  - Changing from mean to median stack complicates computation.
  - Explored further in subsequent lectures.

## Conclusion
- **Detection Techniques**: Essential for identifying and analyzing pulsars.
- **Method Selection**: Choice between mean and median stacking affects results.
- **Future Exploration**: Investigate the impact of different stacking methods on data analysis.




