from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat, GuidedBackprop
from zennit.torchvision import ResNetCanonizer
import torch.nn as nn
import torch
from functools import partial
from zennit.attribution import Gradient, IntegratedGradients, SmoothGrad
from torch.nn import Sequential, Conv2d, ReLU, Linear, Flatten


def prune_by_magnitude(feat, target_sparsity=0.7, max_norm=5):
    batch_size = feat.shape[0]
    num_elements_per_image = feat[
        0
    ].numel()  # Total number of elements in one image (channels x height x width)

    for i in range(batch_size):
        image = feat[
            i
        ].clone()  # Clone the i-th image to avoid modifying the original until pruning is done

        # Flatten the image to 1D
        flattened_image = image.view(-1)

        # Compute the number of elements to prune
        num_elements_to_prune = int(target_sparsity * num_elements_per_image)

        # Get the absolute values of the image and sort them
        sorted_abs_values, _ = torch.sort(torch.abs(flattened_image))

        # Find the threshold corresponding to the desired sparsity
        pruning_threshold = sorted_abs_values[num_elements_to_prune - 1].item()

        # Apply the pruning: Set values where abs(image) <= threshold to 0
        pruned_image = torch.where(
            torch.abs(image) <= pruning_threshold, torch.tensor(0.0), image
        )

        print(
            f"Image {i+1}/{batch_size} - Pruning Threshold: {pruning_threshold}, Sparsity: {target_sparsity * 100}%"
        )
        feat[i] = pruned_image  # Replace the original image with the pruned version

    # Recalculate norm for each image in the batch
    norm = feat.view(feat.shape[0], -1).norm(dim=1, keepdim=True)

    # Clip feat if its norm exceeds max_norm for any batch
    norm = norm.view(feat.shape[0], 1, 1, 1)  # Reshape for broadcasting
    feat = torch.where(norm > max_norm, feat * (max_norm / norm), feat)

    # Final norm calculation after pruning
    norm = feat.view(feat.shape[0], -1).norm(dim=1, keepdim=True)
    print("Clipped Norm Mean:", norm.mean())

    return feat


def sparsify_tensor(tensor, sparsity_percentage):
    """
    Sparsifies a 2D tensor based on the magnitude of pixel values.

    Parameters:
    tensor (torch.Tensor): A 2D tensor with values between 0 and 1.
    sparsity_percentage (float): Desired sparsity percentage (between 0 and 100).

    Returns:
    torch.Tensor: Sparsified tensor.
    """
    # Ensure tensor is 2D and sparsity percentage is between 0 and 100
    if tensor.ndim != 2:
        raise ValueError("Input tensor must be a 2D tensor.")
    if not (0 <= sparsity_percentage <= 100):
        raise ValueError("Sparsity percentage must be between 0 and 100.")

    # Flatten the tensor and calculate the threshold for sparsity
    flattened_tensor = tensor.flatten()
    k = int(len(flattened_tensor) * (sparsity_percentage / 100))

    if k > 0:
        # Sort the values to get the threshold
        sorted_tensor = torch.sort(flattened_tensor).values
        threshold_value = sorted_tensor[k - 1]
    else:
        threshold_value = float("inf")  # No sparsity if k is 0

    # Apply thresholding to sparsify the tensor
    sparsified_tensor = torch.where(
        tensor >= threshold_value, tensor, torch.tensor(0.0, device=tensor.device)
    )

    return sparsified_tensor


def heatmaps_and_labels_zennit(model, loader, method):
    model.cuda()
    model.eval()
    hms = []
    imgs = []
    labels = []
    if method == "Gradient":
        Attr = Gradient(model)
    if method == "IG":
        Attr = IntegratedGradients(model)
    if method == "SmoothGrad":
        Attr = SmoothGrad(x)
    # Iterate over the data loader
    for data in loader:
        x, y = data["img"], data["label"]
        x = x.cuda()
        x.requires_grad_(True)

        # Using Gradient for relevance computation
        with Attr as attributor:
            # Compute output and relevance
            output, relevance = attributor(x)
            relevance = relevance.sum(1)
            # Heatmap and prediction
            pred = output.argmax(dim=1)

            # Append gradients and labels
            for i in range(len(x)):
                hms.append(relevance[i].detach().cpu())  # Store the relevance heatmaps
                imgs.append(x[i].detach().cpu())  # Store the input images
                labels.append(y[i].cpu())  # Store the labels

    return hms, imgs, labels


def heatmaps_and_labels_crp(model, loader):
    model.cuda()
    model.eval()
    hms = []
    imgs = []
    labels = []
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    # GuidedBackProp
    # load CRP toolbox
    attribution = CondAttribution(model)
    # here, each channel is defined as a concept
    cc = ChannelConcept()
    # get layer names of Conv2D and MLP layers
    layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])

    for data in loader:
        try:
            x,y = data["image"], data['label']  # Try to extract the image
        except (TypeError, KeyError):
            try:
                x,y = data["img"], data['label']   # Fallback if "image" key is missing
            except (TypeError, KeyError):
                x,y = data  # Assume data is a tuple, take the first element

        # data = transformed_images
        x.requires_grad_(True)
        # get a conditional attribution for channel 50 in layer features.27 wrt. output 1
        conditions = [{"y": y}]
        attr = attribution(x.cuda(), conditions, composite, record_layer=layer_names)

        # heatmap and prediction
        hm, pred = attr.heatmap, attr.prediction

        # pred_labels = attr.prediction.argmax(dim=1)  # Assuming attr.prediction is the output logits from the model

        # #Check if predictions match the true labels
        # correct_predictions = pred_labels.cuda() == y.cuda()

        # #Append gradients and labels for correct predictions only
        # for i, correct in enumerate(correct_predictions):
        #     if correct.item():
        for i in range(len(x)):
            hms.append(hm[i].detach().cpu())  # .abs()
            imgs.append(x[i].detach().cpu())
            # print(x.grad[i].shape)
            labels.append(y[i].cpu())
    return hms, imgs, labels


def apply_heatmap_mask(images, heatmaps, sparsity_level=60):
    """
    Apply a normalized heatmap to the corresponding images by blending the normalized
    heatmap values with the original images across all color channels.

    Args:
        images (list of torch.Tensor): List of images with shape (3, 32, 32).
        heatmaps (list of torch.Tensor): List of heatmaps with shape (32, 32), corresponding to the images.
        alpha (float): Blending factor to control the strength of the heatmap effect on the images.

    Returns:
        blended_images (list of torch.Tensor): List of images where each pixel has been blended
                                               with the corresponding heatmap value, applied across all channels.
    """
    blended_images = []
    hms = []

    for image, heatmap in zip(images, heatmaps):
        # Normalize the heatm   ap to be between 0 and 1
        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
        normalized_heatmap = sparsify_tensor(
            normalized_heatmap, sparsity_percentage=sparsity_level
        )
        hms.append(normalized_heatmap)
        # inverted_map = 1 - normalized_heatmap
        # Expand the normalized heatmap across the color channels
        expanded_heatmap = normalized_heatmap.unsqueeze(0).repeat(3, 1, 1)
        # expanded_inverted_map = inverted_map.unsqueeze(0).repeat(3,1,1)
        # blended_image = image - image*expanded_inverted_map
        # Blend the image with the expanded heatmap
        blended_image = image * expanded_heatmap
        # blended_image+= torch.empty(blended_image.shape).normal_(0, .15)
        blended_images.append(blended_image)

    return torch.stack(blended_images), torch.stack(hms)


# Example usage:
# images is a list of torch.Tensor, each of shape (3, 32, 32)
# heatmaps is a list of torch.Tensor, each of shape (
# 32, 32)
# Replace these with your actual list of images and heatmaps
# imgs, hms = heatmaps_and_labels(model, loader)
# masked_images, rest_images = partition_image_by_heatmap(imgs, hms)
