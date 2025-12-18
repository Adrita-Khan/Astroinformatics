# Astronomical Data Analysis: Comprehensive Lecture Notes

**Course Overview**: Modern astronomical data analysis combining statistical methods, machine learning, and observational techniques for cosmology and astrophysics research.

---

## Table of Contents

1. [Introduction to Astronomical Data Analysis](#1-introduction)
2. [Python Programming for Astronomy](#2-python-programming)
3. [Statistical Foundations](#3-statistical-foundations)
4. [Data Formats and I/O](#4-data-formats)
5. [Image Data Analysis](#5-image-analysis)
6. [Spectroscopic Data Analysis](#6-spectroscopy)
7. [Time Series Analysis](#7-time-series)
8. [Bayesian Methods](#8-bayesian-methods)
9. [Machine Learning in Astronomy](#9-machine-learning)
10. [Survey Data Analysis](#10-survey-analysis)
11. [Radio Astronomy Data Analysis](#11-radio-astronomy)
12. [Cosmological Data Analysis](#12-cosmological-analysis)

---

## 1. Introduction to Astronomical Data Analysis {#1-introduction}

### 1.1 The Modern Data Landscape

Astronomical data has exploded in volume and complexity:
- **Legacy Surveys**: SDSS, 2MASS, WISE (~petabyte scale)
- **Current Surveys**: DESI, DES, Pan-STARRS, Gaia
- **Next Generation**: Vera Rubin Observatory (LSST), Euclid, Roman Space Telescope
- **Multi-messenger**: Gravitational waves, neutrinos, electromagnetic spectrum

### 1.2 The Data Analysis Pipeline

Standard workflow for astronomical research:

1. **Data Acquisition**: Observational proposals, archival data access
2. **Preprocessing**: Calibration, systematic corrections, quality assessment
3. **Feature Extraction**: Photometry, spectroscopy, morphology
4. **Statistical Analysis**: Parameter estimation, hypothesis testing
5. **Interpretation**: Physical modeling, cosmological inference
6. **Publication**: Visualization, reproducibility, data sharing

### 1.3 Essential Skills

- **Programming**: Python (primary), shell scripting, version control (Git)
- **Statistics**: Probability theory, inference methods, error analysis
- **Domain Knowledge**: Astrophysics, instrumentation, observational techniques
- **Software**: Astropy, NumPy, SciPy, Matplotlib, specialized packages
- **Computing**: Linux/Unix environments, high-performance computing, cloud platforms

---

## 2. Python Programming for Astronomy {#2-python-programming}

### 2.1 Core Libraries

#### NumPy: Numerical Computing
```python
import numpy as np

# Array creation and manipulation
data = np.array([1.2, 3.4, 5.6, 7.8])
mean = np.mean(data)
std = np.std(data)

# Multi-dimensional arrays for images
image = np.zeros((512, 512))  # 512x512 pixel image
image[100:150, 200:250] = 1.0  # Set region to 1
```

#### Astropy: Astronomy-Specific Tools
```python
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.cosmology import Planck18

# Read FITS file
hdulist = fits.open('observation.fits')
data = hdulist[0].data
header = hdulist[0].header

# Physical units
distance = 10 * u.Mpc
distance_km = distance.to(u.km)

# Coordinate transformations
coord = SkyCoord(ra=150.0*u.deg, dec=2.5*u.deg, frame='icrs')
galactic = coord.galactic

# Cosmological calculations
z = 0.5  # redshift
luminosity_distance = Planck18.luminosity_distance(z)
```

#### Matplotlib: Visualization
```python
import matplotlib.pyplot as plt

# Standard plot
plt.figure(figsize=(10, 6))
plt.plot(wavelength, flux, 'k-', linewidth=1)
plt.xlabel('Wavelength (Å)', fontsize=12)
plt.ylabel('Flux (erg/s/cm²/Å)', fontsize=12)
plt.title('Spectrum', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('spectrum.png', dpi=300)

# Image display
plt.figure(figsize=(8, 8))
plt.imshow(image, origin='lower', cmap='viridis', 
           vmin=np.percentile(image, 1), 
           vmax=np.percentile(image, 99))
plt.colorbar(label='Counts')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.savefig('image.png', dpi=300)
```

### 2.2 Best Practices

**Code Organization**:
- Modular functions for reusability
- Clear variable names (avoid single letters except loop indices)
- Comprehensive docstrings
- Type hints for function arguments

**Version Control**:
```bash
# Initialize repository
git init
git add analysis.py
git commit -m "Initial analysis script"

# Branching for experiments
git checkout -b new-feature
git merge new-feature  # After testing
```

**Documentation**:
- Jupyter notebooks for exploratory analysis
- Python scripts for production pipelines
- README files explaining workflow
- Requirements.txt for dependencies

---

## 3. Statistical Foundations {#3-statistical-foundations}

### 3.1 Probability Distributions

#### Gaussian (Normal) Distribution
- **Applications**: Measurement errors, instrumental noise
- **PDF**: \( p(x|\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \)
- **Properties**: Mean = median = mode = \(\mu\), 68-95-99.7 rule

#### Poisson Distribution
- **Applications**: Photon counting, source detection
- **PMF**: \( P(k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!} \)
- **Properties**: Mean = variance = \(\lambda\), approximates Gaussian for large \(\lambda\)

#### Chi-Squared Distribution
- **Applications**: Goodness-of-fit testing, model comparison
- **Relation**: Sum of \(k\) squared standard normal variables
- **Reduced \(\chi^2\)**: \( \chi^2_\nu = \frac{1}{\nu}\sum_{i=1}^{N} \frac{(O_i - E_i)^2}{\sigma_i^2} \), where \(\nu = N - p\) (degrees of freedom)

### 3.2 Error Propagation

For function \( f(x, y) \) with independent errors \(\sigma_x, \sigma_y\):

\[
\sigma_f^2 = \left(\frac{\partial f}{\partial x}\right)^2 \sigma_x^2 + \left(\frac{\partial f}{\partial y}\right)^2 \sigma_y^2
\]

**Examples**:
- Sum/Difference: \( \sigma_{x \pm y} = \sqrt{\sigma_x^2 + \sigma_y^2} \)
- Product: \( \frac{\sigma_{xy}}{xy} = \sqrt{\left(\frac{\sigma_x}{x}\right)^2 + \left(\frac{\sigma_y}{y}\right)^2} \)
- Power: \( \frac{\sigma_{x^n}}{x^n} = |n| \frac{\sigma_x}{x} \)

### 3.3 Hypothesis Testing

#### Frequentist Approach
- **Null hypothesis** (\(H_0\)): Default assumption (e.g., no effect)
- **Alternative hypothesis** (\(H_1\)): What you're testing for
- **p-value**: Probability of observing data at least as extreme as yours under \(H_0\)
- **Significance**: Conventionally reject \(H_0\) if \(p < 0.05\) (5σ in particle physics)

#### Kolmogorov-Smirnov Test
- Tests whether two samples come from the same distribution
- Non-parametric (no assumption about distribution shape)
- Useful for comparing observed data to theoretical models

```python
from scipy.stats import ks_2samp

# Compare two datasets
statistic, pvalue = ks_2samp(data1, data2)
if pvalue < 0.05:
    print("Distributions are significantly different")
```

### 3.4 Maximum Likelihood Estimation (MLE)

Find parameters \(\theta\) that maximize the likelihood function:

\[
\mathcal{L}(\theta | \text{data}) = \prod_{i=1}^{N} p(x_i | \theta)
\]

In practice, minimize negative log-likelihood:

\[
-\ln \mathcal{L} = -\sum_{i=1}^{N} \ln p(x_i | \theta)
\]

```python
from scipy.optimize import minimize

def neg_log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(np.log(1/(sigma*np.sqrt(2*np.pi)) * 
                   np.exp(-(data - mu)**2/(2*sigma**2))))

# Initial guess
initial = [0.0, 1.0]
result = minimize(neg_log_likelihood, initial, args=(data,))
mu_mle, sigma_mle = result.x
```

---

## 4. Data Formats and I/O {#4-data-formats}

### 4.1 FITS Files (Flexible Image Transport System)

Standard format for astronomical data since 1981.

**Structure**:
- **Primary HDU**: Header + Data (image or null)
- **Extension HDUs**: Additional images, tables (binary or ASCII)

**Reading FITS Files**:
```python
from astropy.io import fits

# Open file
hdul = fits.open('galaxy_image.fits')
print(hdul.info())  # List all HDUs

# Access primary data
primary_data = hdul[0].data
primary_header = hdul[0].header

# Important header keywords
exposure_time = primary_header['EXPTIME']
filter_name = primary_header['FILTER']
ra = primary_header['RA']
dec = primary_header['DEC']

# Access table extension
table = hdul[1].data
magnitudes = table['MAG']
redshifts = table['REDSHIFT']

hdul.close()
```

**Writing FITS Files**:
```python
# Create new FITS file
primary_hdu = fits.PrimaryHDU(data=image_array)
primary_hdu.header['OBJECT'] = 'NGC 1234'
primary_hdu.header['EXPTIME'] = 300.0
primary_hdu.header['FILTER'] = 'r'

# Save
primary_hdu.writeto('output.fits', overwrite=True)
```

### 4.2 Catalogs and Tables

**ASCII Tables**:
```python
from astropy.table import Table

# Read various formats
catalog = Table.read('sources.csv', format='csv')
catalog = Table.read('sources.dat', format='ascii')

# Access columns
ra = catalog['RA']
dec = catalog['DEC']
mag = catalog['MAG_AUTO']

# Filtering
bright_sources = catalog[catalog['MAG_AUTO'] < 20]
```

**VO Tables** (Virtual Observatory):
```python
# Common for survey data downloads
votable = Table.read('query_result.xml', format='votable')
```

### 4.3 Data Quality and Validation

**Check for issues**:
```python
# NaN values
n_nans = np.sum(np.isnan(data))

# Infinite values
n_infs = np.sum(np.isinf(data))

# Negative values in images (potential cosmic rays)
n_negative = np.sum(data < 0)

# Basic statistics
print(f"Min: {np.min(data)}, Max: {np.max(data)}")
print(f"Mean: {np.mean(data)}, Median: {np.median(data)}")
print(f"Std: {np.std(data)}")
```

---

## 5. Image Data Analysis {#5-image-analysis}

### 5.1 Image Calibration

#### Bias Correction
Remove electronic offset:
```python
# Combine multiple bias frames
bias_frames = [fits.getdata(f'bias_{i:02d}.fits') for i in range(1, 11)]
master_bias = np.median(bias_frames, axis=0)

# Apply to science frame
science = fits.getdata('science.fits')
science_corrected = science - master_bias
```

#### Dark Current Correction
Remove thermal signal:
```python
# Scale dark to exposure time
master_dark = fits.getdata('master_dark.fits')
dark_exptime = 300  # seconds
science_exptime = 600
dark_scaled = master_dark * (science_exptime / dark_exptime)

science_corrected = science - master_bias - dark_scaled
```

#### Flat Fielding
Correct for pixel-to-pixel sensitivity variations:
```python
# Create normalized flat
master_flat = fits.getdata('master_flat.fits')
master_flat_normalized = master_flat / np.median(master_flat)

# Apply
science_calibrated = (science - master_bias - dark_scaled) / master_flat_normalized
```

### 5.2 Background Estimation

**Simple Method** (for sparse fields):
```python
# Sigma-clipped mean
from astropy.stats import sigma_clipped_stats

mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
background_level = median
```

**Sophisticated Method** (crowded fields):
```python
from photutils.background import Background2D, MedianBackground

# Estimate background in boxes
bkg_estimator = MedianBackground()
bkg = Background2D(image, box_size=(50, 50), 
                   filter_size=(3, 3),
                   bkg_estimator=bkg_estimator)

# Subtract
image_bkg_subtracted = image - bkg.background
```

### 5.3 Source Detection

```python
from photutils.detection import DAOStarFinder

# Detect sources
threshold = 5.0 * std  # 5-sigma threshold
fwhm = 3.0  # pixels
daofind = DAOStarFinder(threshold=threshold, fwhm=fwhm)
sources = daofind(image - background_level)

print(f"Found {len(sources)} sources")
print(sources['xcentroid', 'ycentroid', 'flux'])
```

### 5.4 Aperture Photometry

```python
from photutils.aperture import CircularAperture, aperture_photometry

# Define apertures
positions = list(zip(sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=5.0)

# Perform photometry
phot_table = aperture_photometry(image - background_level, apertures)

# Convert to magnitudes
from astropy.stats import sigma_clipped_stats
flux = phot_table['aperture_sum']
magnitude = -2.5 * np.log10(flux) + zero_point
```

### 5.5 PSF Photometry

For crowded fields where apertures overlap:
```python
from photutils.psf import DAOPhotPSFPhotometry, IntegratedGaussianPRF
from astropy.modeling.fitting import LevMarLSQFitter

# Build PSF model
psf_model = IntegratedGaussianPRF(sigma=2.0)
fitter = LevMarLSQFitter()

# Perform PSF fitting
daogroup = DAOGroup(2.0 * fwhm)
photometry = DAOPhotPSFPhotometry(
    finder=daofind,
    group_maker=daogroup,
    psf_model=psf_model,
    fitter=fitter,
    fitshape=(11, 11),
    aperture_radius=5
)

result = photometry(image - background_level)
```

---

## 6. Spectroscopic Data Analysis {#6-spectroscopy}

### 6.1 Wavelength Calibration

Using arc lamp emission lines:
```python
# Known wavelengths of calibration lines (e.g., Neon, Argon)
known_wavelengths = np.array([5852.49, 6143.06, 6402.25, 6717.04])  # Å
observed_pixels = np.array([245.3, 312.8, 378.2, 450.1])

# Fit polynomial (typically 3rd or 4th order)
wavelength_solution = np.polyfit(observed_pixels, known_wavelengths, deg=3)

# Apply to science spectrum
pixel_array = np.arange(len(science_spectrum))
wavelength_array = np.polyval(wavelength_solution, pixel_array)
```

### 6.2 Sky Subtraction

For long-slit or multi-object spectroscopy:
```python
# Select sky regions (no sources)
sky_region1 = science_2d[10:50, :]  # rows 10-50
sky_region2 = science_2d[400:440, :]

# Median combine
sky_spectrum = np.median(np.vstack([sky_region1, sky_region2]), axis=0)

# Subtract from each row
science_2d_sky_subtracted = science_2d - sky_spectrum[np.newaxis, :]
```

### 6.3 Spectral Extraction

```python
# Extract 1D spectrum from 2D
source_rows = slice(200, 250)  # rows containing source
spectrum_1d = np.sum(science_2d_sky_subtracted[source_rows, :], axis=0)

# Optimal extraction (weighted by S/N)
variance = np.abs(science_2d_sky_subtracted) + readnoise**2
weights = science_2d_sky_subtracted / variance
spectrum_optimal = np.sum(weights[source_rows, :], axis=0)
```

### 6.4 Redshift Measurement

**Cross-correlation Method**:
```python
from scipy.signal import correlate

# Template spectrum (rest frame)
template_wave, template_flux = load_template()

# Cross-correlate
correlation = correlate(observed_flux, template_flux, mode='same')
lag = np.arange(-len(correlation)//2, len(correlation)//2)

# Find peak
max_lag = lag[np.argmax(correlation)]
velocity_shift = max_lag * pixel_scale * c  # km/s

# Redshift
z = velocity_shift / c
```

**Emission Line Fitting**:
```python
from scipy.optimize import curve_fit

def gaussian(x, amp, center, sigma):
    return amp * np.exp(-(x - center)**2 / (2 * sigma**2))

# Fit Hα line
mask = (wavelength > 6550) & (wavelength < 6580)
wave_region = wavelength[mask]
flux_region = flux[mask]

popt, pcov = curve_fit(gaussian, wave_region, flux_region, 
                       p0=[100, 6563, 2])
observed_wavelength = popt[1]
rest_wavelength = 6562.8  # Hα rest wavelength

z = (observed_wavelength - rest_wavelength) / rest_wavelength
```

### 6.5 Spectral Line Measurements

**Equivalent Width**:
\[
EW = \int \frac{F_c - F_\lambda}{F_c} d\lambda
\]

```python
# Continuum fit (polynomial to regions without lines)
continuum_mask = ((wavelength < 6550) | (wavelength > 6580))
continuum_fit = np.polyfit(wavelength[continuum_mask], 
                           flux[continuum_mask], deg=1)
continuum = np.polyval(continuum_fit, wavelength)

# Equivalent width
line_mask = (wavelength > 6550) & (wavelength < 6580)
ew = np.trapz((continuum[line_mask] - flux[line_mask]) / continuum[line_mask],
              wavelength[line_mask])
print(f"Equivalent Width: {ew:.2f} Å")
```

---

## 7. Time Series Analysis {#7-time-series}

### 7.1 Light Curves

**Basic Operations**:
```python
from astropy.time import Time
import astropy.units as u

# Time handling
times_jd = Time(jd_array, format='jd', scale='utc')
times_mjd = times_jd.mjd

# Phase folding
period = 1.5 * u.day
phase = ((times_jd.jd - epoch) % period.value) / period.value

# Bin light curve
from scipy.stats import binned_statistic
bins = np.linspace(0, 1, 20)
binned_flux, bin_edges, _ = binned_statistic(phase, flux, 
                                              statistic='median', bins=bins)
```

### 7.2 Period Finding

#### Lomb-Scargle Periodogram
For unevenly sampled data:
```python
from astropy.timeseries import LombScargle

# Create periodogram
frequency, power = LombScargle(times_mjd, flux, flux_err).autopower()

# Find peak
best_frequency = frequency[np.argmax(power)]
best_period = 1 / best_frequency

print(f"Best period: {best_period:.4f} days")

# False alarm probability
fap = LombScargle(times_mjd, flux, flux_err).false_alarm_probability(power.max())
print(f"False alarm probability: {fap:.4e}")
```

### 7.3 Fourier Analysis

```python
from scipy.fft import fft, fftfreq

# Discrete Fourier Transform
N = len(flux)
sampling_rate = 1 / np.median(np.diff(times_mjd))  # day^-1

fft_values = fft(flux)
fft_freq = fftfreq(N, d=1/sampling_rate)

# Power spectrum
power_spectrum = np.abs(fft_values)**2

# Plot positive frequencies only
positive_freq_mask = fft_freq > 0
plt.plot(fft_freq[positive_freq_mask], power_spectrum[positive_freq_mask])
plt.xlabel('Frequency (day⁻¹)')
plt.ylabel('Power')
```

### 7.4 Transient Detection

**Change Point Detection**:
```python
from scipy.stats import median_abs_deviation

# Rolling statistics
window = 10
rolling_median = pd.Series(flux).rolling(window, center=True).median()
rolling_mad = pd.Series(flux).rolling(window, center=True).apply(median_abs_deviation)

# Detect outliers
threshold = 5.0
outliers = np.abs(flux - rolling_median) > threshold * rolling_mad
transient_times = times_mjd[outliers]
```

---

## 8. Bayesian Methods {#8-bayesian-methods}

### 8.1 Bayes' Theorem

\[
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
\]

Where:
- \(P(\theta | D)\): **Posterior** - probability of parameters given data
- \(P(D | \theta)\): **Likelihood** - probability of data given parameters
- \(P(\theta)\): **Prior** - prior knowledge about parameters
- \(P(D)\): **Evidence** - normalization constant

### 8.2 Prior Selection

**Uninformative Priors**:
- Uniform: \( P(\theta) = \text{const} \) over range
- Jeffreys: \( P(\theta) \propto 1/\theta \) (scale invariant)

**Informative Priors**:
- From previous studies
- Physical constraints (e.g., positive distances)

```python
def log_prior(params):
    """Log prior probability"""
    mu, sigma = params
    # Uniform prior on mu (-10 to 10)
    # Log-uniform prior on sigma (0.1 to 10)
    if -10 < mu < 10 and 0.1 < sigma < 10:
        return -np.log(sigma)  # Jeffreys prior on sigma
    return -np.inf  # Outside bounds
```

### 8.3 MCMC Sampling

**Markov Chain Monte Carlo** samples from posterior distribution.

```python
import emcee

def log_probability(params, x, y, yerr):
    """Log posterior = log prior + log likelihood"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y, yerr)

def log_likelihood(params, x, y, yerr):
    """Gaussian likelihood"""
    model = linear_model(x, params)
    return -0.5 * np.sum(((y - model) / yerr)**2 + np.log(2*np.pi*yerr**2))

# Initialize walkers
nwalkers = 32
ndim = 2  # number of parameters
initial = np.random.randn(nwalkers, ndim)

# Run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                 args=(x_data, y_data, y_err))
sampler.run_mcmc(initial, 5000, progress=True)

# Extract samples (discard burn-in)
samples = sampler.get_chain(discard=1000, flat=True)
```

### 8.4 Posterior Analysis

```python
import corner

# Corner plot
fig = corner.corner(samples, labels=['μ', 'σ'],
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True, title_fmt='.3f')

# Parameter estimates (median and 68% credible interval)
from numpy import percentile
mu_mcmc = percentile(samples[:, 0], [16, 50, 84])
sigma_mcmc = percentile(samples[:, 1], [16, 50, 84])

print(f"μ = {mu_mcmc[1]:.3f} +{mu_mcmc[2]-mu_mcmc[1]:.3f} -{mu_mcmc[1]-mu_mcmc[0]:.3f}")
```

### 8.5 Model Comparison

**Bayesian Information Criterion (BIC)**:
\[
BIC = -2 \ln \mathcal{L}_{\text{max}} + k \ln n
\]

where \(k\) = number of parameters, \(n\) = number of data points.

Lower BIC indicates better model (balances fit quality and complexity).

```python
def calculate_bic(log_likelihood_max, n_params, n_data):
    return -2 * log_likelihood_max + n_params * np.log(n_data)

# Compare models
bic_model1 = calculate_bic(logL1, k1, n)
bic_model2 = calculate_bic(logL2, k2, n)

if bic_model1 < bic_model2:
    print("Model 1 preferred")
```

---

## 9. Machine Learning in Astronomy {#9-machine-learning}

### 9.1 Classification Tasks

**Applications**:
- Galaxy morphology (spiral, elliptical, irregular)
- Stellar classification (spectral types)
- Transient identification (supernovae, AGN, variable stars)
- RFI detection in radio astronomy

**Random Forest Example**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data
X = features  # shape: (n_samples, n_features)
y = labels    # shape: (n_samples,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = clf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### 9.2 Regression Tasks

**Applications**:
- Photometric redshift estimation
- Stellar parameter determination (Teff, log g, [Fe/H])
- Supernova light curve fitting

**Neural Network for Photo-z**:
```python
from tensorflow import keras
from tensorflow.keras import layers

# Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output: redshift
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train, y_train, 
                   validation_split=0.2,
                   epochs=100, 
                   batch_size=32,
                   verbose=1)

# Evaluate
y_pred = model.predict(X_test)
residuals = y_test - y_pred.flatten()
scatter = np.std(residuals)
bias = np.mean(residuals)
print(f"Photo-z scatter: {scatter:.4f}, bias: {bias:.4f}")
```

### 9.3 Convolutional Neural Networks (CNNs)

For image-based tasks (galaxy morphology, gravitational lens detection):

```python
from tensorflow.keras import layers, models

# Build CNN
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train on images
model.fit(images_train, labels_train, 
          epochs=50, 
          validation_data=(images_test, labels_test))
```

### 9.4 Dimensionality Reduction

**Principal Component Analysis (PCA)**:
```python
from sklearn.decomposition import PCA

# Reduce spectral data
pca = PCA(n_components=10)
spectra_reduced = pca.fit_transform(spectra)  # (n_spectra, 10)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

**t-SNE for Visualization**:
```python
from sklearn.manifold import TSNE

# 2D embedding
tsne = TSNE(n_components=2, random_state=42)
embedding = tsne.fit_transform(features)

# Plot
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=5)
plt.colorbar(label='Class')
```

### 9.5 Unsupervised Clustering

**K-Means for Source Clustering**:
```python
from sklearn.cluster import KMeans

# Cluster galaxies by color
colors = catalog[['u-g', 'g-r', 'r-i', 'i-z']].values

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(colors)

# Visualize
plt.scatter(colors[:, 0], colors[:, 1], c=cluster_labels, cmap='viridis', s=1)
plt.xlabel('u - g')
plt.ylabel('g - r')
```

---

## 10. Survey Data Analysis {#10-survey-analysis}

### 10.1 Cross-Matching Catalogs

```python
from astropy.coordinates import SkyCoord, match_coordinates_sky

# Create coordinate objects
coords1 = SkyCoord(ra=catalog1['RA']*u.deg, dec=catalog1['DEC']*u.deg)
coords2 = SkyCoord(ra=catalog2['RA']*u.deg, dec=catalog2['DEC']*u.deg)

# Find matches within 1 arcsecond
idx, sep, _ = match_coordinates_sky(coords1, coords2)
matches = sep < 1.0 * u.arcsec

# Matched catalogs
matched_cat1 = catalog1[matches]
matched_cat2 = catalog2[idx[matches]]

print(f"Matched {np.sum(matches)} sources")
```

### 10.2 Photometric Redshifts

**Template Fitting Method**:
```python
# Load SED templates
templates = load_templates()  # (n_templates, n_wavelengths)

# Observed photometry
observed_flux = catalog[['u', 'g', 'r', 'i', 'z']].values
observed_err = catalog[['u_err', 'g_err', 'r_err', 'i_err', 'z_err']].values
filter_wavelengths = np.array([3543, 4770, 6231, 7625, 9134])  # Å

# Grid search over redshift
z_grid = np.linspace(0, 3, 300)
chi2_grid = np.zeros((len(catalog), len(z_grid), len(templates)))

for i, z in enumerate(z_grid):
    for j, template in enumerate(templates):
        # Redshift template
        template_z = np.interp(filter_wavelengths * (1+z), 
                               template_wavelengths, template_flux)
        # Normalize and compute chi2
        norm = np.sum(observed_flux * template_z / observed_err**2) / \
               np.sum(template_z**2 / observed_err**2)
        chi2_grid[:, i, j] = np.sum(((observed_flux - norm*template_z) / observed_err)**2, axis=1)

# Best-fit redshift
best_idx = np.argmin(chi2_grid.reshape(len(catalog), -1), axis=1)
z_phot = z_grid[best_idx % len(z_grid)]
```

### 10.3 Completeness and Selection Functions

```python
# Magnitude limit determination
def completeness_function(mag, catalog, mag_column='MAG_AUTO'):
    """Fraction of sources recovered as function of magnitude"""
    # Inject artificial sources and recover
    n_bins = 20
    mag_bins = np.linspace(mag.min(), mag.max(), n_bins)
    completeness = np.zeros(n_bins - 1)
    
    for i in range(len(mag_bins) - 1):
        mask = (mag >= mag_bins[i]) & (mag < mag_bins[i+1])
        n_injected = np.sum(mask)
        n_recovered = np.sum(catalog[mag_column][mask] > 0)  # simplified
        completeness[i] = n_recovered / n_injected if n_injected > 0 else 0
    
    return (mag_bins[:-1] + mag_bins[1:]) / 2, completeness

# 50% completeness limit
mag_bins, comp = completeness_function(injected_mags, catalog)
mag_50 = np.interp(0.5, comp, mag_bins)
print(f"50% completeness at mag = {mag_50:.2f}")
```

### 10.4 Luminosity Functions

Schechter function:
\[
\Phi(L) dL = \Phi^* \left(\frac{L}{L^*}\right)^\alpha \exp\left(-\frac{L}{L^*}\right) \frac{dL}{L^*}
\]

```python
from scipy.optimize import curve_fit

def schechter(M, phi_star, M_star, alpha):
    """Schechter luminosity function in magnitudes"""
    L_ratio = 10**(0.4 * (M_star - M))
    return 0.4 * np.log(10) * phi_star * L_ratio**(alpha + 1) * np.exp(-L_ratio)

# Histogram of absolute magnitudes
M_abs = catalog['MAG'] - distance_modulus
bins = np.linspace(-24, -16, 30)
counts, bin_edges = np.histogram(M_abs, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit Schechter function
popt, pcov = curve_fit(schechter, bin_centers, counts, 
                       p0=[1e-3, -21, -1.0])
phi_star, M_star, alpha = popt
print(f"M* = {M_star:.2f}, α = {alpha:.2f}")
```

---

## 11. Radio Astronomy Data Analysis {#11-radio-astronomy}

### 11.1 Visibility Data and Fourier Transforms

Radio interferometers measure Fourier components of the sky brightness:
\[
V(u, v) = \int \int I(l, m) e^{-2\pi i (ul + vm)} dl\, dm
\]

**Imaging via Inverse Fourier Transform**:
```python
from scipy.fft import ifft2, fftshift

# Visibility data: (u, v, V_real, V_imag)
# Grid visibilities onto regular (u,v) grid
uv_grid = grid_visibilities(u, v, V)  # Custom function

# Inverse FFT to get image
image = fftshift(ifft2(uv_grid))
image_cleaned = np.abs(image)  # Amplitude
```

### 11.2 RFI Mitigation

**Statistical Flagging**:
```python
from scipy.stats import kurtosis

# Spectral kurtosis for RFI detection
SK = kurtosis(visibilities, axis=0)  # Along time axis

# Flag high kurtosis (RFI contaminated)
threshold = 3.0
rfi_flags = np.abs(SK - 1) > threshold

# Apply flags
visibilities_clean = visibilities.copy()
visibilities_clean[:, rfi_flags] = np.nan
```

### 11.3 Spectral Line Analysis

**21 cm Hydrogen Line**:
```python
# Convert frequency to velocity
rest_freq = 1420.405751  # MHz (HI rest frequency)
observed_freq = frequencies  # MHz

velocity = 3e5 * (rest_freq - observed_freq) / rest_freq  # km/s

# Integrate over velocity range
velocity_range = (-200, 200)  # km/s
mask = (velocity > velocity_range[0]) & (velocity < velocity_range[1])
integrated_flux = np.trapz(spectrum[mask], velocity[mask])  # Jy km/s

# Column density
N_HI = 1.823e18 * integrated_flux  # cm^-2 (assuming optically thin)
```

### 11.4 Continuum Source Fitting

```python
from astropy.modeling import models, fitting

# 2D Gaussian for unresolved source
gaussian_model = models.Gaussian2D(amplitude=1.0, x_mean=128, y_mean=128,
                                   x_stddev=2, y_stddev=2, theta=0)

# Fit to radio image
fitter = fitting.LevMarLSQFitter()
y, x = np.mgrid[:256, :256]
fitted_model = fitter(gaussian_model, x, y, radio_image)

# Extract properties
flux = fitted_model.amplitude.value
position = (fitted_model.x_mean.value, fitted_model.y_mean.value)
size = (fitted_model.x_stddev.value, fitted_model.y_stddev.value)
pa = fitted_model.theta.value * 180 / np.pi  # degrees
```

---

## 12. Cosmological Data Analysis {#12-cosmological-analysis}

### 12.1 Correlation Functions

Two-point correlation function measures galaxy clustering:
\[
\xi(r) = \langle \delta(x) \delta(x + r) \rangle
\]

```python
from scipy.spatial import cKDTree

def correlation_function(ra, dec, z, bins, cosmology):
    """Compute angular correlation function"""
    # Convert to comoving coordinates
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    distances = cosmology.comoving_distance(z)
    x, y, z_cart = distances * np.cos(coords.dec.rad) * np.cos(coords.ra.rad), \
                   distances * np.cos(coords.dec.rad) * np.sin(coords.ra.rad), \
                   distances * np.sin(coords.dec.rad)
    
    # Build tree for pair counting
    tree = cKDTree(np.column_stack([x, y, z_cart]))
    
    # Count pairs
    DD = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        pairs = tree.query_pairs(bins[i+1])
        DD[i] = len([p for p in pairs if bins[i] <= tree.data[p[0], :] - tree.data[p[1], :] < bins[i+1]])
    
    # Normalize (Landy-Szalay estimator requires random catalog)
    return DD  # Simplified

# Compute
r_bins = np.logspace(0, 2, 20)  # Mpc/h
xi = correlation_function(ra, dec, redshift, r_bins, Planck18)
```

### 12.2 Power Spectrum Estimation

```python
# Convert correlation function to power spectrum via Fourier transform
from scipy.integrate import quad

def xi_to_Pk(r, xi, k):
    """Convert correlation function to power spectrum"""
    P = np.zeros_like(k)
    for i, k_val in enumerate(k):
        integrand = lambda r_val: r_val**2 * np.interp(r_val, r, xi) * \
                                  np.sin(k_val * r_val) / (k_val * r_val)
        P[i] = 4 * np.pi * quad(integrand, r.min(), r.max())[0]
    return P

k_values = np.logspace(-2, 0, 50)  # h/Mpc
power_spectrum = xi_to_Pk(r_bins[:-1], xi, k_values)
```

### 12.3 Cosmological Parameter Fitting

**Distance-Redshift Relation**:
```python
from scipy.optimize import minimize

def chi2_cosmology(params, z_obs, mu_obs, mu_err):
    """Chi-squared for cosmological parameters"""
    Om, OL = params
    # Compute distance modulus for each redshift
    cosmo = FlatLambdaCDM(H0=70, Om0=Om, Ode0=OL)
    mu_theory = cosmo.distmod(z_obs).value
    return np.sum(((mu_obs - mu_theory) / mu_err)**2)

# Fit to supernova data
result = minimize(chi2_cosmology, x0=[0.3, 0.7], 
                 args=(sne_redshift, sne_distance_modulus, sne_errors))
Om_best, OL_best = result.x
print(f"Ωm = {Om_best:.3f}, ΩΛ = {OL_best:.3f}")
```

### 12.4 Redshift-Space Distortions

Account for peculiar velocities in redshift measurements:
\[
s = r + \frac{v_{\parallel}}{H(z)}
\]

```python
# Multipole expansion of correlation function
def xi_multipoles(s, mu, xi_s_mu):
    """Compute monopole and quadrupole"""
    from scipy.special import legendre
    
    # Integrate over mu = cos(angle)
    xi_0 = np.trapz(xi_s_mu, mu) / 2  # Monopole (ℓ=0)
    xi_2 = 5/2 * np.trapz(legendre(2)(mu) * xi_s_mu, mu)  # Quadrupole (ℓ=2)
    
    return xi_0, xi_2

# Extract growth rate f
# Theoretical: ξ₂/ξ₀ ≈ (4/3)β where β = f/b
```

### 12.5 CMB Analysis (Simplified)

**Angular Power Spectrum**:
```python
import healpy as hp

# CMB temperature map (HEALPix format)
cmb_map = hp.read_map('cmb_map.fits')

# Compute power spectrum
cl = hp.anafast(cmb_map, lmax=2000)
ell = np.arange(len(cl))

# Plot
Dl = ell * (ell + 1) * cl / (2 * np.pi)  # Convert to D_ℓ
plt.plot(ell, Dl * 1e12)  # μK²
plt.xlabel('Multipole ℓ')
plt.ylabel('D_ℓ [μK²]')
plt.xscale('log')
```

---

## Appendix A: Common Astronomical Units

### Distance
- **Parsec (pc)**: 1 pc = 3.086 × 10¹⁶ m = 3.26 light-years
- **Astronomical Unit (AU)**: 1 AU = 1.496 × 10¹¹ m (Earth-Sun distance)

### Brightness
- **Magnitude**: \( m_1 - m_2 = -2.5 \log_{10}(F_1/F_2) \)
- **Absolute Magnitude**: Apparent magnitude at 10 pc
- **Distance Modulus**: \( \mu = m - M = 5 \log_{10}(d/10\,\text{pc}) \)

### Redshift
- **Cosmological**: \( z = \frac{\lambda_{\text{obs}} - \lambda_{\text{rest}}}{\lambda_{\text{rest}}} = \frac{a_0}{a(t)} - 1 \)
- **Velocity**: \( v = cz \) (for \(z \ll 1\))

---

## Appendix B: Software Packages

### Essential Python Libraries
- **Astropy**: Core astronomical utilities
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Visualization
- **Pandas**: Tabular data manipulation
- **scikit-learn**: Machine learning
- **emcee**: MCMC sampling
- **corner**: Posterior visualization

### Specialized Tools
- **Photutils**: Photometry (aperture, PSF)
- **Specutils**: Spectroscopy
- **HEALPix**: All-sky maps (CMB, surveys)
- **CASA**: Radio interferometry
- **IRAF/PyRAF**: Legacy reduction (being phased out)
- **Source Extractor**: Source detection in images
- **TOPCAT**: Interactive catalog analysis

### Computing Platforms
- **Jupyter Notebooks**: Interactive analysis
- **Google Colab**: Cloud-based Python
- **SciServer**: Astronomy-specific cloud platform
- **Astro Data Lab**: NOAO data access and analysis

---

## Appendix C: Best Practices

### Reproducible Research
1. **Version control**: Track all code changes with Git
2. **Documentation**: README files, docstrings, comments
3. **Environment management**: conda/pip requirements files
4. **Data provenance**: Record data sources, versions, processing steps
5. **Notebooks**: Use for exploration; scripts for production
6. **Testing**: Unit tests for critical functions

### Data Management
1. **Backup**: Multiple copies (local + cloud)
2. **Organization**: Logical directory structure
3. **Metadata**: Document observation parameters, calibration
4. **Archives**: Deposit final products in repositories (Zenodo, VizieR)

### Publication Standards
1. **Figures**: 300+ dpi for journals, vector formats (PDF, EPS)
2. **Tables**: Machine-readable formats (FITS, CSV)
3. **Code sharing**: GitHub/GitLab repositories
4. **Data sharing**: Virtual Observatory compliance when possible

---

## Further Reading

### Textbooks
- **Statistics**: Feigelson & Babu, "Modern Statistical Methods for Astronomy"
- **Data Analysis**: Wall & Jenkins, "Practical Statistics for Astronomers"
- **Python**: VanderPlas, "Python Data Science Handbook"
- **Machine Learning**: Ivezić et al., "Statistics, Data Mining, and Machine Learning in Astronomy"

### Online Resources
- **Astropy Tutorials**: https://learn.astropy.org
- **AstroML**: https://www.astroml.org
- **STScI Notebooks**: https://spacetelescope.github.io/notebooks/
- **Scipy Lecture Notes**: https://scipy-lectures.org

### Courses
- Coursera: "Data-driven Astronomy"
- edX: "Analyzing the Universe"
- YouTube: "Python for Astronomers" series

---



*These notes provide a foundation for modern astronomical data analysis. Adapt techniques to your specific research needs and always validate results with domain expertise.*
