import numpy as np
import torch
import torch.nn as nn

floattype = torch.cuda.FloatTensor

class TorchROIPool(object):

    def __init__(self, output_size, scaling_factor):
        """ROI max pooling works by dividing the hxw RoI window into an HxW grid of 
           approximately size h/H x w/W and then max-pooling the values in each
           sub-window. Pooling is applied independently to each feature map channel.
        """
        self.output_size = output_size
        self.scaling_factor = scaling_factor

    def _roi_pool(self, features):
        """Given scaled and extracted features, do channel wise pooling
        to return features of fixed size self.output_size, self.output_size

        Args:
            features (np.Array): scaled and extracted features of shape
            num_channels, proposal_width, proposal_height
        """

        num_channels, h, w = features.shape

        w_stride = w/self.output_size
        h_stride = h/self.output_size

        res = torch.zeros((num_channels, self.output_size, self.output_size))
        res_idx = torch.zeros((num_channels, self.output_size, self.output_size))
        for i in range(self.output_size):
            for j in range(self.output_size):
                
                # important to round the start and end, and then conver to int
                w_start = int(np.floor(j*w_stride))
                w_end = int(np.ceil((j+1)*w_stride))
                h_start = int(np.floor(i*h_stride))
                h_end = int(np.ceil((i+1)*h_stride))

                # limiting start and end based on feature limits
                w_start = min(max(w_start, 0), w)
                w_end = min(max(w_end, 0), w)
                h_start = min(max(h_start, 0), h)
                h_end = min(max(h_end, 0), h)

                patch = features[:, h_start: h_end, w_start: w_end]
                max_val, max_idx = torch.max(patch.reshape(num_channels, -1), dim=1)
                res[:, i, j] = max_val
                res_idx[:, i, j] = max_idx

        return res, res_idx

    def __call__(self, feature_layer, proposals):
        """Given feature layers and a list of proposals, it returns pooled
        respresentations of the proposals. Proposals are scaled by scaling factor
        before pooling.

        Args:
            feature_layer (np.Array): Feature layer of size (num_channels, width,
            height)
            proposals (list of np.Array): Each element of the list represents a bounding
            box as (w,y,w,h)

        Returns:
            np.Array: Shape len(proposals), channels, self.output_size, self.output_size
        """

        batch_size, num_channels, _, _ = feature_layer.shape

        # first scale proposals based on self.scaling factor 
        scaled_proposals = torch.zeros_like(proposals)

        # the rounding by torch.ceil is important for ROI pool
        scaled_proposals[:, 0] = torch.ceil(proposals[:, 0] * self.scaling_factor)
        scaled_proposals[:, 1] = torch.ceil(proposals[:, 1] * self.scaling_factor)
        scaled_proposals[:, 2] = torch.ceil(proposals[:, 2] * self.scaling_factor)
        scaled_proposals[:, 3] = torch.ceil(proposals[:, 3] * self.scaling_factor)

        res = torch.zeros((len(proposals), num_channels, self.output_size,
                        self.output_size))
        res_idx = torch.zeros((len(proposals), num_channels, self.output_size,
                        self.output_size))
        for idx in range(len(proposals)):
            proposal = scaled_proposals[idx]
            # adding 1 to include the end indices from proposal
            extracted_feat = feature_layer[0, :, proposal[1].to(dtype=torch.int8):proposal[3].to(dtype=torch.int8)+1, proposal[0].to(dtype=torch.int8):proposal[2].to(dtype=torch.int8)+1]
            res[idx], res_idx[idx] = self._roi_pool(extracted_feat)

        return res

if __name__ == "__main__":
    
    from torchvision.ops.roi_pool import RoIPool

    # create feature layer, proposals and targets
    num_proposals = 10
    feat_layer = torch.randn(1, 64, 32, 32)

    proposals = torch.zeros((num_proposals, 4))
    proposals[:, 0] = torch.randint(0, 16, (num_proposals,))
    proposals[:, 1] = torch.randint(0, 16, (num_proposals,))
    proposals[:, 2] = torch.randint(16, 32, (num_proposals,))
    proposals[:, 3] = torch.randint(16, 32, (num_proposals,))

    print('proposals=', proposals.size(), proposals.type())

    my_roi_pool_obj = TorchROIPool(3, 2**-1)
    roi_pool1 = my_roi_pool_obj(feat_layer, proposals)

    print('Roi pool 1=', roi_pool1.size())

    roi_pool_obj = RoIPool(3, 2**-1)
    roi_pool2 = roi_pool_obj(feat_layer, [proposals])
    print('Roi pool 2=', roi_pool2.size())