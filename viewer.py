import matplotlib.pyplot as plt
import matplotlib
import cv2
matplotlib.use('TkAgg')

def show_image(img = None, title=['image'], type_plot = 1):

    if isinstance(img, list):

        if type_plot == 1:

            for i in range(len(img)):
                plt.subplot(1, len(img), i+1)
                if len(img[i].shape) < 3:
                    plt.imshow(img[i], cmap='gray', vmin=0, vmax=255)
                else:
                    plt.imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
                if len(img) != len(title):
                    plt.title(title[0])
                else:
                    plt.title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

        if type_plot == 2:

            qtd_images = len(img)
            qtd_titles = len(title)
            fig, axes = plt.subplots(nrows=1, ncols=qtd_images, figsize=(32, 18), sharex=True, sharey=True)
            ax = axes.ravel()

            for i in range(qtd_images):
                if len(img[i].shape) < 3:
                    ax[i].imshow(img[i], cmap='gray', vmin=0, vmax=255)
                else:
                    ax[i].imshow(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
                if qtd_images != qtd_titles:
                    ax[i].set_title(title[0])
                else:
                    ax[i].set_title(title[i])
                plt.xticks([]), plt.yticks([])
            plt.show()

    else:
        if len(img.shape) < 3:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)

        plt.title(title[0])
        plt.xticks([]), plt.yticks([])
        plt.show()

def show_image_opencv(image):

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_video(video):

    title = 'Video'
    frames = []  # for storing the generated images
    fig = plt.figure()

    if len(video[0].shape) < 3:
        type_image = 'GRAY'
    else:
        type_image = 'COLOR'

    for i in range(len(video)):
        if type_image == 'COLOR':
            frames.append([plt.imshow(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB), vmin=0, vmax=255, animated=True)])
        else:
            frames.append([plt.imshow(video[i], cmap='gray', vmin=0, vmax=255, animated=True)])
        plt.title(title)
        plt.xticks([]), plt.yticks([])
    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True, repeat_delay=1000)
    plt.show()

if __name__ == '__main__':

    image = cv2.imread('/home/anderson/Downloads/gerber_stencil.png')

    show_image([image])