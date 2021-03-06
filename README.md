Overview

This is a program that attempts to solve Sudoku puzzles that have been extacted from an image.

How it works:

Through various image processing techniques combined with optical character recognition (OCR), this program will try to extract the digits from a Sudoku puzzle. Once the digits are extracted, it will attempt to solve it. The process works as follows:

1. Find the main grid for the puzzle
2. Find the four corners associated with this grid
3. Perform a homology to extract just the puzzle from the image
4. Perform a form of thresholding if necessary
5. Extract each cell from the puzzle
6. Perform OCR to determine the digits
7. Solve the puzzle

What is included:

1. Test data to train a support vector machine used for optical character recognition
2. Various test puzzles with the ground truth values used for comparison.

Required libraries:

1. OpenCV
2. Numpy
3. PyLab
4. LibSVM
5. Python Imaging Library (PIL)
6. Scipy
7. Execution

To run the program:

type python imsudoku.py

Results:

The average accuracy of extracting the initial digits was 86%.

For any questions, contact rohitvasudevan23@gmail.com

