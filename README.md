# ImageSegmentationUsingKMeans

Image segmentation using KMeans

## Run 
`python Image_Segmentation_using_kmeans.py --k 3 --input_file input/burma.jpg`

- **k** : It is the required no of clusters to be formed. Default is 3. Higher the no, more is the time required for each iteration and thus for convergence.

- **input_file** : The relative path of the desired input file

NOTE :: 
- Output images and gif will be created in output folder.
- Images are automatically being resized to reduce the time taken.

## RESULTS
March comes in like a lion Image
![Image](input/march-comes-in-like-a-lion-wallpapers.jpg) ![Segmentation Output](ouput/segmentation.gif)
