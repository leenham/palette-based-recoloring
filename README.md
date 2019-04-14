Palette-based Photo Recoloring
===========
source palette_recolor_venv/bin/activate

python main.py

v1.0
===========
implementation based on <Palette-based Photo Recoloring> using python.
A naive User Interface by PyQt.
Implement the following functions:
1. init function for accelerating K-means.
2. K-means (K is defaultly set to be 5)
3. recolor the photo if user changes one or several palettes.(using RBF functions)

The initialization process costs 5~6s, and update process costs less than 1s.
