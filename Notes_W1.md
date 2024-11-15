# WEEK-1 NOTES

### Pulsars
- **Definition**: Pulsars are highly magnetized, rotating neutron stars that emit beams of electromagnetic radiation.
- **Formation**: They are formed from supernova explosions, where a massive star collapses, leading to high-frequency neutron beams and intense magnetic and gravitational fields.
- **Two-Star Collision**: When two stars collide, they may also trigger a supernova, which can form a pulsar.

### Astronomical Image Formats
- **FITS Format**: Astronomical images are commonly saved in `.fits` (Flexible Image Transport System) format.
- **Software for Viewing**: Tools like DS9 and Aladin are used to open and analyze these FITS files.

### Flux Density and Units
- **Unit**: The unit of flux density is the Jansky (Jy).
  - **Conversion**: \( 1 \, \text{Jy} = 1 \times 10^{-26} \, \text{W/m}^2 \text{Hz} \)
  - \( \text{W} = \text{watt}, \, \text{m} = \text{meter}, \, \text{Hz} = \text{Hertz} \)

### Sample Sources and Flux Densities
| **Source**              | **Flux Density (Jy)** |
|-------------------------|-----------------------|
| Mobile Phone            | 110,000,000          |
| Sun at 10 GHz           | 4,000,000            |
| Milky Way at 10 GHz     | 2,000                |
| Crab Pulsar at 1.4 GHz  | 0.01                 |

### Storage Calculation for Images
- **Question**: How much memory is required to store 600,000 images at 200 x 200 px resolution?
  - **Calculation**:
    - 1 image = 200 x 200 px = 40,000 pixels
    - 1 pixel = 8 bytes
    - Total memory = \( 600,000 \times 40,000 \times 8 \, \text{bytes} = 192 \, \text{GB} \)

  - **Alternative Compression**:
    - If the images don't contain critical data, they can be resized to 50 x 50 px.
    - This would reduce storage needs by a factor of 4, requiring only 16 GB.

---

**END OF WEEK-1 NOTES**
