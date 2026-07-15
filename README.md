# DRUID
[![Run tests](https://github.com/RhysAlfShaw/DRUID/actions/workflows/pytest.yaml/badge.svg)](https://github.com/RhysAlfShaw/DRUID/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/RhysAlfShaw/DRUID/graph/badge.svg?token=C4KD4C6IXA)](https://codecov.io/gh/RhysAlfShaw/DRUID)

DRUID is a general-purpose source finder for optical and radio images written in Python, applicable to a broad range of scenarios. 

DRUID relies on the use of persistent homology to find sources and nested components within an image. This information is then processed as described in Shaw et al. (in prep). 

Currently, DRUID uses the [`cripser`](https://github.com/shizuo-kaji/CubicalRipser_3dim) library to calculate the persistence of homology groups within 2D data.

## Versions

This is the newly parallelized version of DRUID, featuring improved background data handling. The version of DRUID used in Shaw et al. (2025) can be found in the [releases]().

### Notes on the new version
- DRUID's architecture has been significantly updated to allow for highly efficient parallelization.
- Pandas DataFrames have been replaced with Polars, drastically improving compute speed and memory management.
- GPU usage has been removed, as it was incompatible with the new parallel strategy.

These changes have increased DRUID's speed by roughly 10x. This improvement stems mostly from the new processing architecture and the switch to Polars, with further gains driven by parallelization. To get an idea of its new performance, check out the scaling plot below. This benchmark reflects the processing of an optical image with 0.1" resolution over a 0.57 deg² field of view, containing around 100,000 sources.

![DRUID Performance Scaling](docs/assets/DRUID_performance_scaling.png)

## Installation

Currently, the best way to use DRUID is to clone this repository and install it along with its dependencies:

```bash
pip install .
```

You can then verify the installation by running:

```python
from DRUID import sf
```

### Note for Apple Silicon Users
`cripser` does not provide compiled binaries for Apple Silicon, so you will need to compile the library locally. This can typically be done with the following command:

```bash
pip install -U git+[https://github.com/shizuo-kaji/CubicalRipser_3dim](https://github.com/shizuo-kaji/CubicalRipser_3dim)
```

Any installation errors at this stage will likely stem from the version of CMake or the C compilers you have installed. See the [CubicalRipser_3dim repository](https://github.com/shizuo-kaji/CubicalRipser_3dim) for further details on required compilers.

## Using DRUID

To run DRUID, follow these steps:

1. **Initialize the `sf` (source finding) object:**
```python
findmysource = sf(image=image, image_path=None, mode='optical', area_limit=5, header=header)
```

2. **Define the background:**
```python
findmysource.set_background(detection_threshold=5, analysis_threshold=2, mode='rms')
```

3. **Find and deblend sources using Persistent Homology:**
```python
findmysource.phsf()
```

4. **Characterize the sources:** Now that we have a list of sources and a hierarchy of nested components, we can characterize them and measure their properties.
```python
findmysource.source_characterising(use_gpu=False)
```

To explore how DRUID can be used in practice, check out the example notebooks where we demonstrate several of DRUID's functions. *(Coming soon: based on the analysis in Shaw et al., in prep).*

### Saving the Catalogue
To save the output catalogue along with the contours, you should use the `save_catalogue()` function, as this will properly serialize the object. To correctly open the catalogue again, use `open_catalogue()` after initializing the `sf` class.

## Bugs & Issues

Please report any bugs or issues you encounter while using DRUID on this repository's [Issues](#) page. Thank you!

## Further Application & Development

If you want to extend DRUID's capabilities—whether that means adding new functionality or improving what is already implemented—feel free to submit a pull request or email me at [rhys.shaw@bristol.ac.uk](mailto:rhys.shaw@bristol.ac.uk) to discuss.

## Acknowledgements

If you use DRUID for your research, please cite: 
> [Shaw et al. 2025](https://doi.org/10.1093/rasti/rzaf006)