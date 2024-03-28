# DRUID
[![Run tests](https://github.com/RhysAlfShaw/DRUID/actions/workflows/pytest.yaml/badge.svg)](https://github.com/RhysAlfShaw/DRUID/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/RhysAlfShaw/DRUID/graph/badge.svg?token=C4KD4C6IXA)](https://codecov.io/gh/RhysAlfShaw/DRUID)

DRUID is a general purpose source finder for optical and radio images written in `python`. That can be appied in a broad range of situations. It comes some GPU acceleration of key compute functions, making processing of large sources faster.

DRUID relies on the use of Persistent homology to find sources and nested components from within the image. This information is then processed as described in Shaw et al (in prep). 

Currently DRUID uses the `criper` (https://github.com/shizuo-kaji/CubicalRipser_3dim) library to calculate the persistence of homology groups within the 2d data.

# Installation

To use DRUID currently the best way is to clone this repository and install it with its dependencies.

```bash
pip install .
```

You can then test if it is working with.

```python
from DRUID import sf
```
## Note for Apple Silicon

For Apple Silcon Users! cripser does not provide compiled binaries for apple silicon. So you need to compile the library locally. This should be simple with the command:

```bash
pip install -U git+https://github.com/shizuo-kaji/CubicalRipser_3dim
```

Any installation error from here will likely be from the the version of CMAKE or the C compilers you have installed. See https://github.com/shizuo-kaji/CubicalRipser_3dim for further details on required compilers.

## Using the GPU functionality

To use the GPU functions that DRUID offers you need to install `cupy` [https://cupy.dev/].
This is not done in the normal requirements as it requires access to a Nvidia GPU with Cuda. And hence requires cuda tool kit to be installed on you machine.

If you have sucessfully install cupy then you can use `GPU=True`.

# Using DRUID

To use DRUID you need to do the following steps.

1. Initailise the sf (source finding) object.
```python
findmysource = sf(image=image,image_path=None, mode='optical',area_limit=5,GPU=True, header=header)
```
2. Define the background.
```python
findmysource.set_background(detection_threshold=5,analysis_threshold=2,mode='rms')
```
3. Find and Deblend sources with Persistent Homology.
```python
findmysources.phsf()
```
4. Now we have a list of sources and a hierachy of nested components, we can charaterise them and measure some properties.

```pythons
findmysources.source_charaterising(use_gpu=False)
```

To explore how DRUID can be used Check out the example notebooks where we demonstraight some of DRUIDs functions. (This is incoming, will be based on analysis in Shaw et al in prep)

## Saving the catalogue.
To save the output catalogue with the contours you should use the ```save_catalogue()``` function. As this will properly save the object. To correctly open the catalogue again use ```open_catalogue()``` after initlising the sf class.

# Bugs/issues

Please report any bug or issues using DRUID to this repositories issue page. Thank you.

# Further application/developement

If you want to increase the functionality, whether thats adding additional functionality to improving what is already implemented, feel free to submit a pull request or email me (rhys.shaw@bristol.ac.uk) to discuss.

# Acknowledgements

If you use DRUID for your research please cite: (link to DRUID paper)