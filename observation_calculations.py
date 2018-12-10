mport os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



def create_observation_probabilities(patch_image_directory, context_patch, loc, pw):
    print('done')
    # open the context image
    # context = Image.open(context_image)
    # get context image scaled down
    # patchR, patchC, _ = patch_images[0].shape
    # context = context.resize((patchC, patchR), Image.ANTIALIAS)
    # context = np.array(context)
     # now for each small patch in context, find the best stylized patch
    # output = np.zeros(context.shape)
    # for r in range(1, patchR-smallest_pw, stride):
    #     print("on row ", r, " of ", patchR-smallest_pw)
    #     for c in range(1, patchC-smallest_pw, stride):
    #         context_patch = context[r:r+smallest_pw, c:c+smallest_pw, :]
    #         best_texture_index = None
    #         best_texure_dist = float('inf')
            # print("\nfor the patch tl at ", r, ", ", c)
            # print(context_patch.shape)
    # look at each possible patch texture
    distances = []
    denom = 0
    for i, texture in enumerate(patch_images):
        text_patch = texture[loc[0]:loc[0]+pw, loc[1]:loc[1]+pw, :]
        dist = evaluate_patch_distance(text_patch, context_patch)
        denom += dist
        distances.append(dist)
                 # print("dist: ", dist)
                # print(text_patch.shape)
                 # if dist < best_texure_dist:
                 #    best_texture_index = i
                 #    best_texure_dist = dist
             # use best texture patch in final output
            # output[r:r+smallest_pw, c:c+smallest_pw, :] = patch_images[best_texture_index][r:r+smallest_pw, c:c+smallest_pw, :]
    #create a k x 1 matrix that favors smaller distances
    prob_matrix = []
    for dist in distances:
        prob_matrix.append([1-(dist/denom)])

    return prob_matrix
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
    if len(sys.argv) < 5:
        print("Expected: create_pyramid.py <patch_image_directory> <context_image> <stride>")
        sys.exit(2)
    print('Starting combining')
    create_observation_probabilities(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
