import os
import sys
from PIL import Image
import numpy as np
# from matplotlib import pyplot as plt
from pyugm.model import Model
from pyugm.factor import DiscreteFactor
from pyugm.factor import DiscreteBelief
from pyugm.infer_message import LoopyBeliefUpdateInference
from pyugm.infer_message import FloodingProtocol, LoopyDistributeCollectProtocol
# import seaborn


def combine_best_patches(patch_image_directory, context_image, prediction_fn, start_stride=1):
    """
    This function just looks at the context image in a set patch size and 
        constructs a new image by taking the best patch of that size from 
        K pre-constructed stylized images that were run on varying patch sizes
    inputs
        context_image- the path to the context image (must be full path/wrt this directory)
        stride- steps taken between patches, default 1
            Ideally want some overlap here
    """
    # open the context image
    context = Image.open(context_image)

    # get smallest pw and images as np arrays
    smallest_pw, patch_images = get_stylized_images(patch_image_directory)

    # get context image scaled down so that it's the same size as the stylized imgs
    patchR, patchC, _ = patch_images[0].shape
    context = context.resize((patchC, patchR), Image.ANTIALIAS)
    context = np.array(context)

    print(context.shape)

    pyramid_folder = 'pyramid_output/'
    if not os.path.exists(pyramid_folder):
        os.makedirs(pyramid_folder)

    inner_folder = pyramid_folder + prediction_fn + '_folder/'
    if not os.path.exists(inner_folder):
        os.makedirs(inner_folder)

    stride = start_stride
    while stride <= smallest_pw:
        # now for each small patch in context, find the best stylized patch
        num_overlaps = np.zeros((context.shape[0], context.shape[1]))
        output = np.zeros(context.shape)

        # track which K-style gets used
        rrange = range(0, patchR-smallest_pw+1, stride)
        crange = range(0, patchC-smallest_pw+1, stride)
        # this will be a mapping of context-img r, c to the k-index of the img
        k_usage = np.array([[None for i in crange] for j in rrange])
        kr = 0
        for r in rrange:
            if r%100==0:
                print("on row ", r, " of ", patchR-smallest_pw)
            kc = 0
            for c in crange:
                context_patch = context[r:r+smallest_pw, c:c+smallest_pw, :]
                best_texture_index = None
                best_texture_dist = float('inf')

                # look at each possible patch texture
                for i, texture in enumerate(patch_images):
                    text_patch = texture[r:r+smallest_pw, c:c+smallest_pw, :]
                    dist = evaluate_patch_distance(text_patch, context_patch)

                    if dist < best_texture_dist:
                        best_texture_index = i
                        best_texture_dist = dist

                # use best texture patch in final output
                output[r:r+smallest_pw, c:c+smallest_pw, :] += patch_images[best_texture_index][r:r+smallest_pw, c:c+smallest_pw, :]
                num_overlaps[r:r+smallest_pw, c:c+smallest_pw] += 1
                k_usage[kr][kc] = [[r, c], best_texture_index]
                kc += 1
            kr += 1
        
        print(num_overlaps.shape)
        print(k_usage.shape)

        # save the k usage info
        np.save("%s%s_stride_%d_pyramid_kchoice_data" % (inner_folder, prediction_fn, stride), k_usage)

        # since the patches can overlap, avg out the additions TODO: maybe make this weighted
        for i in range(context.shape[0]):
            for j in range(context.shape[1]):
                if num_overlaps[i][j] == 0:
                    continue # this happens when stride doesn't match up with pw
                # need to make sure it doesn't integer divide
                output[i][j]/=(1.*num_overlaps[i][j])
                
        print("\nSanity check, all values of output OK?")
        print(not (output > 255).any())
        print(not (output < 0).any())

        output = np.uint8(output)
        im = Image.fromarray(output)
        im.save(inner_folder+ prediction_fn + 'stride_%d_pyramid_output.png' % stride)
        stride += 1

def lbp_combine_best_patches(patch_image_directory, context_image, prediction_fn, stride=1, max_iters=1):
    # open the context image
    context = Image.open(context_image)

    # get smallest pw and images as np arrays
    smallest_pw, patch_images = get_stylized_images(patch_image_directory)

    # get context image scaled down so that it's the same size as the stylized imgs
    patchR, patchC, _ = patch_images[0].shape
    context = context.resize((patchC, patchR), Image.ANTIALIAS)
    context = np.array(context)

    evidence, factors = construct_graph(patch_images, context, smallest_pw, stride)
    model = Model(factors)

    # Get some feedback on how inference is converging by listening in on some of the label beliefs.
    def reporter(infe, orde):
        print('{:3}'.format(orde.total_iterations))

    order = FloodingProtocol(model, max_iterations=max_iters)
    inference = LoopyBeliefUpdateInference(model, order, callback=reporter)
    inference.calibrate(evidence)

    lbf_folder = 'lbf_output/'
    if not os.path.exists(lbf_folder):
        os.makedirs(lbf_folder)

    inner_folder = lbf_folder + prediction_fn + '_folder/'
    if not os.path.exists(inner_folder):
        os.makedirs(inner_folder)

    K = len(patch_images)
    rrange = range(0, patchR-smallest_pw+1, stride)
    crange = range(0, patchC-smallest_pw+1, stride)
    num_r = len(rrange) # number of patches vertically
    num_c = len(crange) # number of patches horizontally

    ff_labels = [[None for i in range(num_c)] for j in range(num_r)]
    ff_r = 0
    for r in rrange:
        ff_c = 0
        for c in crange:
            variable_name = 'label_{}_{}'.format(r, c)

            # first factor is the context-style factor tha we want
            label_factor = inference.get_marginals(variable_name)[0]

            # save the actual patch location to make it easier to remap them later on
            ff_labels[ff_r][ff_c] = [[r, c], label_factor.normalized_data]
            ff_c+=1
        ff_r+=1

    # save the labels so they can be easily reused
    ff_labels = np.array(ff_labels)
    np.save("%s%s_patchw_%d_first_factor_label_data" % (inner_folder, prediction_fn, smallest_pw), ff_labels)

def get_stylized_images(patch_image_directory):
    """
    This is a helper function to format the stylized images in the given directory into 
    a list of np arrays and find the smallest patch width used in the given imgs
    inputs
        patch_image_directory- directory where stylized imgs are located. 
            They must be named with the convention *_patch_NUM_stylized.*
    outputs
        smallest_pw- the smallest patch width used in the imgs in the directory
        patch_images- a list of the imgs represented as np arrays
    """
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
        sep = fn.split('_')
        patch_width = int(sep[-2])

        if patch_width < smallest_pw:
            smallest_pw = patch_width
        patch_images.append(np.array(Image.open(patch_image_directory+file)))
    return smallest_pw, patch_images

def construct_graph(patch_images, context_image, pw, stride):
    """
    Drawn heavily from this example notebook http://nbviewer.jupyter.org/github/dirko/pyugm/blob/master/examples/Loopy%20belief%20propagation%20example.ipynb
    on how to use the pyugm package

    This will create a graph with observation factors (create_ovservation_comps)
    and label factors (create_neighbor_matrix) s.t. we reward using patches
    from one of the K style imgs that is close to our source image while also 
    rewarding smoothness between the chosen patches (this is done through the neighbor label factors)

    inputs
        patch_image_directory- directory where stylized imgs are located. 
        context_image- the context image as a np array
        pw- patch width
        stride- step size between each patch
    outputs
        evidence- context values associated with each observed variable
        factors- the lines on our graph; they relate the style patch to the context patch
            and relate style patches to other style patches (for neighbors)
    """
    evidence = {}
    factors = []

    K = len(patch_images)
    source_rows = context_image.shape[0]
    source_cols = context_image.shape[1]

    # Add observation factors
    for r in range(0, source_rows-pw+1, stride):
        for c in range(0, source_cols-pw+1, stride):
            # do this for each patch in the source image
            # call create_observation_comps to make the parameters
            label_variable_name = 'label_{}_{}'.format(r, c)
            observation_variable_name = 'obs_{}_{}'.format(r, c)

            print("\nadded label %s and observation %s" % (label_variable_name, observation_variable_name))
            print("these two are linked by a factor")

            observation_params = create_oberservation_comps(patch_images, context_image, (r,c), pw)
            factors.append(DiscreteFactor([(label_variable_name, K), (observation_variable_name, 256)], parameters=observation_params))
            evidence[observation_variable_name] = context_image[r:r+pw, c:c+pw, :] 
            
    # Add label factors
    # for each patch location in a texture image
    # create a node for it and get the 4 neighbor matrices from the helper function
    # then create the neighbors-factors if appropriate
    # as said below, maybe just start with doing down and right
    for r in range(0, source_rows-pw+1, stride):
        for c in range(0, source_cols-pw+1, stride):
            variable_name = 'label_{}_{}'.format(r, c)
            neighbor_params, neighbor_locs = create_neighbor_matrix(patch_images, (r,c), pw, stride)

            print("\nLooking at label %s" % variable_name)

            # for each valid neighbor, create a factor
            for np, nl in zip(neighbor_params, neighbor_locs):
                if np is None or nl is None:
                    continue
                r, c = nl
                neighbor_name = 'label_{}_{}'.format(r, c)

                print("adding neighbor ", neighbor_name)

                factors.append(DiscreteFactor([(variable_name, K), (neighbor_name, K)], parameters=np))
    return evidence, factors

def create_oberservation_comps(patch_images, context_img, loc, pw):
    """
    inputs
        patch_images- list of pre-stylized images of length K (given in shape (H, W, ...))
        context_img- context image (the truth)
        loc- top left corner tuple of patch to be evaluated in format (r, c)
        pw- patch width
    outputs
        observation_comps- Kx1 size matrix with the compatabilities of each 
            K patch-img option vs the original context img patch
    """
    assert(len(patch_images) > 0)
    assert(patch_images[0].shape == context_img.shape)

    K = len(patch_images)
    observation_comps = np.zeros((K))
    context_patch = context_img[loc[0]:loc[0]+pw, loc[1]:loc[1]+pw, :]
    for i in range(K):
        patch_img = patch_images[i]
        text_patch = patch_img[loc[0]:loc[0]+pw, loc[1]:loc[1]+pw, :]
        dist = evaluate_patch_distance(text_patch, context_patch)
        observation_comps[i] = compatability(dist)
    return observation_comps

def create_neighbor_matrix(patch_images, base_loc, pw, stride):
    """
    inputs
        patch_images- list of pre-stylized images of length K (given in shape (H, W, ...))
        base_loc- top left location tuple (r, c) of center patch
        pw- patch width
        stride- step between patches
    outputs
        neighbor_mat- Up to 2 KxK matrix with "compatability" values between
            every pair of patches in base_loc and a neighboring loc from K different images
            In each matrix, the BASE loc will be represented by the rows while NEIGHBOR will be cols
            NOTE: rn only does 4 neighbors (u, d, l, r) TODO maybe do all 8 including diagonals?
        neighbors- the tl locations of the four neighbors
    """
    assert len(patch_images) > 0

    # create neighbor locs
    # note: we only need to check bottom and right neighbors because
    # we start top left and move right then down 
    # this way we don't create redundant factors
    neighbors = []
    if base_loc[0] + stride + pw - 1 < len(patch_images[0]): # if there's room below 
        neighbors.append((base_loc[0]+stride, base_loc[1]))
    else:
        neighbors.append(None)
    if base_loc[1] + stride + pw - 1 < len(patch_images[0][0]): # if there's room to the right
        neighbors.append((base_loc[0], base_loc[1]+stride))
    else:
        neighbors.append(None)

    # create each neighbor mat one by one, using None if the neighbor doesn't exist
    K = len(patch_images)
    neighbor_matrices = []
    for nloc in neighbors:
        print(nloc)
        if nloc is None:
            neighbor_matrices.append(None)
            continue
        neighbor_mat = np.zeros((K, K))
        for i in range(K):
            img_one = patch_images[i]
            patch_one = img_one[base_loc[0]:base_loc[0]+pw, base_loc[1]:base_loc[1]+pw, :]
            for j in range(K):
                img_two = patch_images[j]
                patch_two = img_two[nloc[0]:nloc[0]+pw, nloc[1]:nloc[1]+pw, :]
                dist = evaluate_patch_distance(patch_one, patch_two)
                neighbor_mat[i][j] = compatability(dist)
        neighbor_matrices.append(neighbor_mat)

    # sanity check? 
    for nm in neighbor_matrices:
        if nm is not None:
            assert nm.shape[0] == K and nm.shape[1] == K
    return neighbor_matrices, neighbors

def evaluate_patch_distance(patch_one, patch_two):
    """
    inputs
        patch_one- some square matrix representing a patch
        patch_two- "" of same size as patch_one
    outputs
        distance- normalized distance between two patches
    """
    assert len(patch_one) == len(patch_two)
    pw = len(patch_one)
    dist = np.linalg.norm(patch_one-patch_two)
    normalized_dist = dist/(pw*pw)
    return normalized_dist

def compatability(dist):
    """
    inputs
        dist- normalized euclidean distance
    outputs
        compatability- inverse of distance with a distance of 0 being rounded to 1e-6 to prevent nan
    """
    if dist == 0:
        return np.log(1/1e-6)
    return np.log(1/dist)

if __name__=='__main__':
    if len(sys.argv) < 7:
        print("Expected: create_pyramid.py <use_lbf> <patch_image_directory> <context_image> <prediction_filename> <stride> <max_iters>")
        sys.exit(2)

    if int(sys.argv[1]) == 0:
        print('Starting combining using best patches')
        combine_best_patches(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        print('Starting combining using lbf')
        lbp_combine_best_patches(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))