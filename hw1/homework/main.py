import time
import numpy as np
import cv2 as cv
import math as mt
import matplotlib.pyplot as plt

def generic_2d_convol(img, kernel):
    convol_img = np.zeros(shape = (len(img) - len(kernel), len(img[0] - len(kernel[0]))));

    for i in range(len(img) - len(kernel)):
        for j in range(len(img[0]) - len(kernel[0])):
            convol_img[i,j] = sum([img[i + x,j + y] * kernel[x, y] for x in range(len(kernel)) for y in range(len(kernel[0]))])

    return convol_img


def sharpness_filter(img): 
    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    b, g, r = cv.split(img)

    b_ch_sharp = generic_2d_convol(img = np.array(b), kernel = sharpen).astype(int)
    r_ch_sharp = generic_2d_convol(img = np.array(r), kernel = sharpen).astype(int)
    g_ch_sharp = generic_2d_convol(img = np.array(g), kernel = sharpen).astype(int)

    return cv.merge([r_ch_sharp, g_ch_sharp, b_ch_sharp])


def blur_filter(img):
    blur = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125],
        [0.0625, 0.125, 0.0625]
    ])

    b, g, r = cv.split(img)

    b_ch_blur = generic_2d_convol(img = np.array(b), kernel = blur).astype(int)
    r_ch_blur = generic_2d_convol(img = np.array(r), kernel = blur).astype(int)
    g_ch_blur = generic_2d_convol(img = np.array(g), kernel = blur).astype(int)

    return cv.merge([r_ch_blur, g_ch_blur, b_ch_blur])


def outline_filter(img):
    outline = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    b, g, r = cv.split(img)

    b_ch_outline = generic_2d_convol(img = np.array(b), kernel = outline).astype(int)
    r_ch_outline = generic_2d_convol(img = np.array(r), kernel = outline).astype(int)
    g_ch_outline = generic_2d_convol(img = np.array(g), kernel = outline).astype(int)

    return cv.merge([r_ch_outline, g_ch_outline, b_ch_outline])


def plot_image(img): 
    plt.figure(figsize=(6, 6))
    plt.imshow(img);


def plot_3_images(plts):
    _, axis = plt.subplots(1, 3, figsize=(18, 6))
    axis[0].imshow(plts[0])
    axis[1].imshow(plts[1])
    axis[2].imshow(plts[2])


def benchmark(func, args):
    start_time = time.time();
    res = func(*args);
    end_time = time.time();

    elapsed_time = end_time - start_time

    return [res, round(elapsed_time * 1000)];


def benchmark_single_ch_convol(channel, kernel_size):
    rand_kernel = np.random.random((kernel_size, kernel_size))
    result = benchmark(generic_2d_convol, (np.array(channel), rand_kernel));

    print("Execution time in ms for kernel of size #{size} = {time}".format(size = kernel_size, time = result[1]));


def main():

    bender_img = cv.imread('/Users/catalin/Documents/computer-vision/w1/bender.png')
    bojack_img = cv.imread('/Users/catalin/Documents/computer-vision/w1/bojack.jpeg')
    a_img = cv.imread('/Users/catalin/Documents/computer-vision/w1/a.jpg')
    b_img = cv.imread('/Users/catalin/Documents/computer-vision/w1/b.jpg')
    
    # Separate b,g,r channels of the original image
    b, g, r = cv.split(bender_img);

    # 1) Using numpy indexing implement a generic 2D convolution operation working on single channel inputs. 
    # Consider using both indexing and block matrix operations to compute the convolution output. 
    # Compare the runtime against the dimensions of the convolution filter, keeping the same image dimensions. 
    # Plot the result of the convolution on each channel using subfigures from matplotlib.
    # Response: Single run returned the following results for sizes of 3,5,8,11: 2290ms, 6016ms, 14831ms, 27526ms

    for i in range(3, 13, 3):
        benchmark_single_ch_convol(b, i)

    # 2) Using the generic convolution operation, implement different image gradients and different smoothing filters. 
    # Plot the filtered output against the image input. Why are the results becoming more/less sharp, after each filtering operation? 

    plot_3_images([sharpness_filter(bender_img), blur_filter(bender_img), outline_filter(bender_img)])

    # 3) Given a noisy image, apply Otsu binarization on the noisy image and a gaussian filtered output of the same image. 
    # Discuss the results given the particularities of Otsu's algorithm. 
    # Discuss about the two phases of the iterative procedure, and how the procedure converges to the given solution.

    ret3,th3 = cv.threshold(bender_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU);

    plt.show()
    
    
if __name__ == '__main__':
    main()