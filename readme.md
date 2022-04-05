# ROAD EXTRACTION

This is a python application that extracts road from inputted images using canny edge detection, contour-feature extraction and a pixel-based elimination.

# Sample data: 

Sample data are within data/ extracted from google images.

# Installation guide:

Clone this repo:
```bash 
git clone https://github.com/marsyaaaaal/RoadDetection
```

# Requirements:
Python >= 3.7 https://www.python.org/downloads/

Opencv

Numpy

```bash 
pip install opencv-python
```


```bash 
pip install numpy
```


# Run road.py::
```bash 
python road.py -i data/road.jpg
```

## Best for road images that aren't noisy (clean version of road) and lesser road details, since it relies on pixel-based extraction