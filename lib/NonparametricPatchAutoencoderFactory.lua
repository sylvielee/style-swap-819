require 'nn'
local NonparametricPatchAutoencoderFactory = torch.class('NonparametricPatchAutoencoderFactory')

function NonparametricPatchAutoencoderFactory.buildAutoencoder(target_img, patch_size, stride, shuffle, normalize, interpolate)
    local nDim = 3
    assert(target_img:nDimension() == nDim, 'target image must be of dimension 3.')

    patch_size = patch_size or 3
    stride = stride or 1

    local type = target_img:type()
    local C = target_img:size(nDim-2)
    local target_patches = NonparametricPatchAutoencoderFactory._extract_patches(target_img, patch_size, stride, shuffle)
    local npatches = target_patches:size(1)

    print("img size")
    print(target_img:size(1), target_img:size(2), target_img:size(3))
    print("looking at patches")
    print(npatches)
    print(target_patches:nDimension())
    print(target_patches:size(1), target_patches:size(2), target_patches:size(3), target_patches:size(4))

    local conv_enc, conv_dec = NonparametricPatchAutoencoderFactory._build(patch_size, stride, C, target_patches, npatches, normalize, interpolate)
    return conv_enc, conv_dec
end

function NonparametricPatchAutoencoderFactory._build(patch_size, stride, C, target_patches, npatches, normalize, interpolate)
    -- for each patch, divide by its L2 norm.
    local enc_patches = target_patches:clone()
    for i=1,npatches do
        enc_patches[i]:mul(1/(torch.norm(enc_patches[i],2)+1e-8))
    end

    ---- Convolution for computing the semi-normalized cross correlation ----
    local conv_enc = nn.SpatialConvolution(C, npatches, patch_size, patch_size, stride, stride):noBias()
    conv_enc.weight = enc_patches
    conv_enc.gradWeight = nil
    conv_enc.accGradParameters = __nop__
    conv_enc.parameters = __nop__

    if normalize then
        -- normalize each cross-correlation term by L2-norm of the input
        local aux = conv_enc:clone()
        aux.weight:fill(1)
        aux.gradWeight = nil
        aux.accGradParameters = __nop__
        aux.parameters = __nop__
        local compute_L2 = nn.Sequential()
        compute_L2:add(nn.Square())
        compute_L2:add(aux)
        compute_L2:add(nn.Sqrt())

        local normalized_conv_enc = nn.Sequential()
        local concat = nn.ConcatTable()
        concat:add(conv_enc)
        concat:add(compute_L2)
        normalized_conv_enc:add(concat)
        normalized_conv_enc:add(nn.CDivTable())
        normalized_conv_enc.nInputPlane = conv_enc.nInputPlane
        normalized_conv_enc.nOutputPlane = conv_enc.nOutputPlane
        conv_enc = normalized_conv_enc
    end

    ---- Backward convolution for one patch ----
    local conv_dec = nn.SpatialFullConvolution(npatches, C, patch_size, patch_size, stride, stride):noBias()
    conv_dec.weight = target_patches
    conv_dec.gradWeight = nil
    conv_dec.accGradParameters = __nop__
    conv_dec.parameters = __nop__

    -- normalize input so the result of each pixel location is a
    -- weighted combination of the backward conv filters, where
    -- the weights sum to one and are proportional to the input.
    -- the result is an interpolation of all filters.
    if interpolate then
        local aux = nn.SpatialFullConvolution(1, 1, patch_size, patch_size, stride, stride):noBias()
        aux.weight:fill(1)
        aux.gradWeight = nil
        aux.accGradParameters = __nop__
        aux.parameters = __nop__

        local counting = nn.Sequential()
        counting:add(nn.Sum(1,3))           -- sum up the channels
        counting:add(nn.Unsqueeze(1,2))     -- add back the channel dim
        counting:add(aux)
        counting:add(nn.Squeeze(1,3))
        counting:add(nn.Replicate(C,1,2))   -- replicates the channel dim C times.

        interpolating_conv_dec = nn.Sequential()
        local concat = nn.ConcatTable()
        concat:add(conv_dec)
        concat:add(counting)
        interpolating_conv_dec:add(concat)
        interpolating_conv_dec:add(nn.CDivTable())
        interpolating_conv_dec.nInputPlane = conv_dec.nInputPlane
        interpolating_conv_dec.nOutputPlane = conv_dec.nOutputPlane
        conv_dec = interpolating_conv_dec
    end

    return conv_enc, conv_dec
end

function NonparametricPatchAutoencoderFactory._extract_patches(img, patch_size, stride, shuffle)
    local nDim = 3
    assert(img:nDimension() == nDim, 'image must be of dimension 3.')
    local kH, kW = patch_size, patch_size
    local dH, dW = stride, stride
    local patches = img:unfold(2, kH, dH):unfold(3, kW, dW)
    local n1, n2, n3, n4, n5 = patches:size(1), patches:size(2), patches:size(3), patches:size(4), patches:size(5)
    patches = patches:permute(2,3,1,4,5):contiguous():view(n2*n3, n1, n4, n5)

    -- print("patch in 2 steps")
    -- print(img:unfold(2, kH, dH):nDimension())
    -- print(img:unfold(2, kH, dH):size(1), img:unfold(2, kH, dH):size(2), img:unfold(2, kH, dH):size(3), img:unfold(2, kH, dH):size(4))
    -- print("part 2")
    -- print(img:unfold(2, kH, dH):unfold(3, kW, dW):nDimension())
    -- print(n1, n2, n3, n4, n5)

    -- print("testing cutting half of patches")
    -- local i = 1
    --     x = torch.LongTensor((n2*n3)/4):apply(function(x)
    --         i = i + 1
    --         return i
    --     end)
    -- patches = patches:index(1, torch.LongTensor({1,2,3,4,5}))
    -- print(patches:nDimension())
    -- print(patches:size(1), patches:size(2), patches:size(3), patches:size(4))

    if shuffle then
        local shuf = torch.randperm(patches:size(1)):long()
        patches = patches:index(1,shuf)
    end

    return patches
end

function __nop__()
    -- do nothing
end
