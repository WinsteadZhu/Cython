# Cython
Here are the codes for the Assignment 3 of course [Advanced Programming for Language Technologists](https://studentportalen.uu.se/portal/portal/uusp/student/student-course?uusp.portalpage=true&toolMode=studentUse&entityId=142786) at Uppsala University, which optimises the mandelbrot zoom code via Cython.

The steps followed are:
1. Use cython code to speed up the original python code for generating such a mandelbrot zoom which is very slow.
2. Parallelize the rendering of frames on several CPU cores.
3. Generate a 500 frame video at a 1000x1000 pixel resolution.
