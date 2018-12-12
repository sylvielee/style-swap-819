import os
import sys
from PIL import Image
import numpy as np

def analyze_k_usage(filename, original_fn, patch_dir, output_name):
    if not os.path.exists(patch_dir):
        raise Exception("%s directory cannot be found" % patch_dir)
    
    patch_images, patch_img_names = get_patch_sources(patch_dir)

    # get context image scaled down so that it's the same size as the stylized imgs
    context = Image.open(original_fn)
    patchR, patchC, _ = patch_images[0].shape
    context = context.resize((patchC, patchR), Image.ANTIALIAS) 
    context = np.array(context)

    print(patch_images[0].shape)
    print(context.shape)

    # need to specify encoding bc python2 encodes in ascii
    # and latin1 is a superset of ascii
    k_usage = np.load(filename, encoding='latin1')

    print(k_usage.shape)

    # expect 2D data (approach 1), 3D data (square im), or 4D data (approach 2)
    if len(k_usage.shape) < 2 or len(k_usage.shape) > 4:
        raise Exception("Expected 2D, 3D, or 4D input")

    data_dir = os.path.dirname(filename) + "/"
    create_figures(k_usage, context, patch_images, patch_img_names, data_dir, output_name)

def create_figures(k_usage, content, patch_images, patch_img_names, output_dir, name, pw=3):
    stats = {ke: 0 for ke in range(len(patch_images))}

    split_styles = np.array([np.zeros(content.shape) for i in patch_images])
    split_style_counts = np.array([np.zeros((content.shape[0], content.shape[1])) for i in patch_images])
    print("\nsplit styles shape")
    print(split_styles.shape)

    color_coded = np.ones(content.shape)*255
    colors = get_colors(len(patch_images))
    print("\ncolor coded shape")
    print(color_coded.shape)

    output = np.zeros(content.shape)
    num_counts = np.zeros((content.shape[0], content.shape[1]))
    print("\noutput shape")
    print(output.shape)

    k_partial_folder = output_dir + name + "_split_style_figures/"
    if not os.path.exists(k_partial_folder):
        os.mkdir(k_partial_folder)

    for i in range(k_usage.shape[0]):
        for j in range(k_usage.shape[1]):
            # entry: [[originalr, originalc], k-index OR [len K]]
            entry = k_usage[i][j]
            orig_r = int(entry[0][0])
            orig_c = int(entry[0][1])

            k_index = -1
            if type(entry[1]) == list or type(entry[1]) == np.ndarray:
                k_index = int(np.argmax(np.array(entry[1])))
            else:
                k_index = entry[1]

            # log which k it was for stats later
            stats[k_index] += 1

            # make actual output
            output[orig_r:orig_r+pw, orig_c:orig_c+pw, :] += patch_images[k_index][orig_r:orig_r+pw, orig_c:orig_c+pw, :]
            num_counts[orig_r:orig_r+pw, orig_c:orig_c+pw] += 1

            # add to the appropriate split style img and count
            split_styles[k_index, orig_r:orig_r+pw, orig_c:orig_c+pw, :] += content[orig_r:orig_r+pw, orig_c:orig_c+pw, :]
            split_style_counts[k_index, orig_r:orig_r+pw, orig_c:orig_c+pw] += 1

            # color appropriate section (for now just overrides with new colors)
            color_coded[orig_r:orig_r+pw, orig_c:orig_c+pw, :] = colors[k_index]

    # normalize stat counts and save
    total = sum(stats.values())
    for ke in stats:
        stats[ke] /= (1.*total)
    with open(output_dir+name+"_stats.txt", 'w') as f:
        f.write("Fraction each k style file was used")
        edist = np.linalg.norm(output-content)
        f.write("Euclidean dist: %.3f" % edist)
        for ke in stats:
            f.write("\n%d, %s: %.4f" % (ke, patch_img_names[ke], stats[ke]))
            f.write("\n\tcolor RGB %d, %d, %d" % (colors[ke][0], colors[ke][1], colors[ke][2]))
    print("wrote stats!")

    print("\nSanity check, all values of output OK?")
    print(not (output > 255).any())
    print(not (output < 0).any())

    # avg actual output and save
    for i in range(content.shape[0]):
        for j in range(content.shape[1]):
            if num_counts[i][j] == 0:
                continue # this happens when stride doesn't match up with pw
            # need to make sure it doesn't integer divide
            output[i][j]/=(1.*num_counts[i][j])
    output = np.uint8(output)
    im = Image.fromarray(output)
    im.save(output_dir + name + '_actual_output.png')
    print("wrote actual output!")

    # avg split styles and save
    for ki in range(len(split_styles)):
        for i in range(content.shape[0]):
            for j in range(content.shape[1]):
                if split_style_counts[ki][i][j] == 0:
                    split_styles[ki][i][j] = 255 # want white background not black
                    continue
                split_styles[ki][i][j] /= (1.*split_style_counts[ki][i][j])
        img = np.uint8(split_styles[ki])
        img = Image.fromarray(img)
        img.save(k_partial_folder + ('%s_partial.png'%patch_img_names[ki]))
    print("wrote partials!")

    # save color coded img
    color_coded = np.uint8(color_coded)
    im = Image.fromarray(color_coded)
    im.save(output_dir + name + '_color_coded.png')
    print("done!")

def get_colors(n):
    colors = np.array([[255, 153, 153], [255, 204, 153], [255, 255, 153], [204, 253, 153], [153, 255, 153], [153, 255, 204],
        [153, 255, 255], [153, 204, 255], [153, 153, 255], [204, 153, 255], [255, 153, 255], [255, 153, 204]])
    idx = np.random.choice(len(colors), n)
    return colors[idx]

def get_patch_sources(patch_image_directory):
    """
    This is a helper function to format the stylized images in the given directory into 
    a list of np arrays and return their filenames
        patch_image_directory- directory where stylized imgs are located. 
            They must be named with the convention *_patch_NUM_stylized.*
    outputs
        patch_images- a list of the imgs represented as np arrays
        patch_names- associated filenames
    """
    # extract all the patch images and their patch widths
    patch_images = []
    patch_names = []
    im_exts = set(['.png', '.jpg'])
    for file in os.listdir(patch_image_directory):
        fn, ext = os.path.splitext(file)
        if ext not in im_exts:
            print('skipped file ', fn)
            continue
        patch_images.append(np.array(Image.open(patch_image_directory+file)))
        patch_names.append(fn)
    return patch_images, patch_names

def flatten_k_usage(k_usage):
    """
    we expect k_usage to look like HxWxK where the last dimension
        is an array of len K whose argmax indicates which K-style's patch was used
    """
    flat_k_usage = np.zeros((k_usage.shape[0], k_usage.shape[1]))
    for i in range(k_usage.shape[0]):
        for j in range(k_usage.shape[1]):
            flat_k_usage = np.argmax(k_usage[i][j])
    return flat_k_usage

if __name__=='__main__':
    if len(sys.argv) < 5:
        print("Expected: evaluate_k_usage.py <data_filename> <original_image> <patch_image_directory> <output_name>")
        sys.exit(2)
    analyze_k_usage(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
