import numpy as np
import torch
from torch.nn.modules.module import Module


class PruningModule(Module):
    DEFAULT_PRUNE_RATE = {
        'conv1': 47,
        'conv2': 12,
        'conv3': 3,
        'conv4': 1,
        'conv5': 1,
        'fc1': 0.05,
        'fc2': 0.005,
        'fc3': 1
    }

    def _prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################
        weights = module.weight.data.cpu().numpy()
        prune_weights = np.where(abs(weights)<threshold, 0, weights)
        module.weight.data = torch.from_numpy(prune_weights).to(module.weight.device)

    def prune_by_percentile(self, q=DEFAULT_PRUNE_RATE):

        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the (100 - q)th percentile
        # 	of absolute W as the threshold, and then set the absolute weights less than threshold to 0 , and the rest
        # 	remain unchanged.
        ########################

        for name, param in self.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy()
                alive = weights[np.nonzero(weights)]
                percentile_value = np.percentile(abs(alive), (100-q[name.split('.')[0]]))
                weight_dev = param.device
                new_mask = np.where(abs(weights) < percentile_value, 0, weights)
                param.data = torch.from_numpy(new_mask).to(weight_dev)
                print(f'Pruning with threshold : {percentile_value:.4f} for layer {name}')


    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################

            if name in ['fc1', 'fc2', 'fc3', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold:.4f} for layer {name}')
                self._prune(module, threshold)