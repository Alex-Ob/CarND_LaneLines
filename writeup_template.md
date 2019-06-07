# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps.
First, I has set the color threshold for masking
Second, I define the bounding polynom (really, it must be defined by extrinsic calibration data).
Then I combine these two mask to apply for detected edges.
For Canny detector I have stayed on parameters which were defined in demo video.
After getting edges, I apply combined mask to contours found and then use Hough transform with some parameters.
Last, I draw line segments on source frames.

I couldn't yet modify algorithm for join segment because I have got a problem with video recording during the classroom session: video file doesn't been rewrite and previous fragment remained there.

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when horizontal lines appears in the ROI. To prevent this case angle of line slope may be defined and selected in some range.

Another shortcoming could be that linear contours of the cars or trucks may appear in front of camera.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to check line angle and sort all line segments with clustering into 2 groups
Another potential improvement could be change linear segment model to quadratic (or more).
