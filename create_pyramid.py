import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def create_pyramid(patch_image_directory, context_image, stride):
    print('done')

    # open the context image
    context = Image.open(context_image)

    # extract all the patch images and their patch widths
    patch_images = []
    smallest_pw = float('inf')
    im_exts = set(['.png', '.jpg'])
    for file in os.listdir(patch_image_directory):
        fn, ext = os.path.splitext(file)
        if ext not in im_exts:
            print('skipped file ', fn)
            continue
        # find the patch size from filename
        # assumes convention *_patch_NUM_stylized.*
        sep = fn.split('_')
        patch_width = int(sep[-2])

        # print(fn)
        # print('patch num ', patch_width)

        if patch_width < smallest_pw:
            smallest_pw = patch_width

        patch_images.append(np.array(Image.open(patch_image_directory+file)))

    print("SMALLEST")
    print(smallest_pw)

    # in shape (rows, cols, 3)
    # print("check im dims")
    # print(patch_images[0].shape)

    # get context image scaled down
    patchR, patchC, _ = patch_images[0].shape
    context = context.resize((patchC, patchR), Image.ANTIALIAS)
    context = np.array(context)

    # now for each small patch in context, find the best stylized patch
    output = np.zeros(context.shape)

    for r in range(1, patchR-smallest_pw, stride):
        print("on row ", r, " of ", patchR-smallest_pw)
        for c in range(1, patchC-smallest_pw, stride):
            context_patch = context[r:r+smallest_pw, c:c+smallest_pw, :]
            best_texture_index = None
            best_texure_dist = float('inf')
            # print("\nfor the patch tl at ", r, ", ", c)
            # print(context_patch.shape)

            # look at each possible patch texture
            for i, texture in enumerate(patch_images):
                text_patch = texture[r:r+smallest_pw, c:c+smallest_pw, :]
                dist = evaluate_patch_distance(text_patch, context_patch)

                # print("dist: ", dist)
                # print(text_patch.shape)

                if dist < best_texure_dist:
                    best_texture_index = i
                    best_texure_dist = dist

            # use best texture patch in final output
            output[r:r+smallest_pw, c:c+smallest_pw, :] = patch_images[best_texture_index][r:r+smallest_pw, c:c+smallest_pw, :]

    plt.imshow(output)
    plt.show()

    output = np.uint8(output)
    im = Image.fromarray(output)
    im.save('pyramid_output.png')

    return output


def evaluate_patch_distance(texture_patch, context_patch):
    """
    inputs
        texture_patch- some square matrix representing a patch on a texture image
        context_path- "" on the context image
    outputs
        distance- normalized distance between texture and context patches
    """
    pw = len(texture_patch)
    dist = np.linalg.norm(texture_patch-context_patch)
    normalized_dist = dist/(pw*pw)
    return normalized_dist

if __name__=='__main__':
    if len(sys.argv) < 4:
        print("Expected: create_pyramid.py <patch_image_directory> <context_image> <stride>")
        sys.exit(2)

    print('Starting combining')
    create_pyramid(sys.argv[1], sys.argv[2], int(sys.argv[3]))