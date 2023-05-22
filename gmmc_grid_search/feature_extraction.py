import torch.nn.functional as F
import torch.nn as nn
import re
import torch
from tqdm import tqdm 


def layer_names(network):
    num_layers = len(list(network.named_modules())) - 1
    for i, m in enumerate(network.named_modules()):
        mo = re.match('.+($|\n)', m[1].__repr__())
        print('Layer {}: {}: {}'.format(num_layers - i, m[0], mo.group(0).strip()))



def extract_features(taskID, feature_extractor, data_loader, 
                     fine_class=1, coarse_class=2, device=0, 
                     num_to_generate=None, use_raw_images=False, raw_image_transform=None):
    """
    Extract image features and put them into arrays.
    :param feature_extractor: network with hooks on layers for feature extraction
    :param data_loader: data loader of images for which we want features (images, labels, item_ixs)
    :param fine_class: target classification label position in array
    :param coarse_class: task target label position in array
    :param num_to_generate: number of samples to extract features from 
    """
    feature_extractor.model.eval()
    # feature_extractor.model.base.train(False)
    feature_extractor.base_freeze = True
    for param in feature_extractor.model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        # allocate space for features and labels
        features_dict={}
        if raw_image_transform is not None:
            tf = raw_image_transform
        # put features and labels into arrays
        num_gen = 0
        for batch_ix, batch in enumerate(data_loader):
            
            batch_x=batch[0]
            batch_y=batch[fine_class]
            batch_y_c=batch[coarse_class]
            num_gen+=batch_x.shape[0]


            if raw_image_transform is not None:
                batch_feats = tf.apply(batch_x)
            else:
                batch_feats = batch_x

            _,batch_feats = feature_extractor(batch_feats.to(device), taskID)

            if use_raw_images:
                batch_feats['image'] = batch_x

            if batch_ix==0:
                for layer in feature_extractor.layer_names:
                    features_dict[layer] = batch_feats[layer].cpu().type(torch.float32)
                if use_raw_images:
                    features_dict['image'] = batch_feats['image'].cpu().type(torch.float32)
                    
                target_labels = batch_y.long()
                target_coarse_labels = batch_y_c.long()
            else:
                
                for layer in feature_extractor.layer_names:
                    features_dict[layer] = torch.cat((features_dict[layer], batch_feats[layer].cpu().type(torch.float32)), dim=0)
                if use_raw_images:
                    features_dict['image'] = torch.cat((features_dict['image'], batch_feats['image'].cpu().type(torch.float32)), dim=0)

                target_labels = torch.cat((target_labels, batch_y.long()), dim=0)
                target_coarse_labels = torch.cat((target_coarse_labels, batch_y_c.long()), dim=0)

            if num_to_generate is not None:
                if num_gen>=num_to_generate:
                    for layer in feature_extractor.layer_names:
                        features_dict[layer] = features_dict[layer][:num_to_generate,...]
                    if use_raw_images:
                        features_dict['image']=features_dict['image'][:num_to_generate,...]
                    target_labels=target_labels[:num_to_generate]
                    target_coarse_labels=target_coarse_labels[:num_to_generate]
                    break
        
    return features_dict, target_labels, target_coarse_labels


class NetworkLatents():
    def __init__(self, model: nn.Module, layer_names, pool_factors=None):
        self.layer_names = layer_names
        self.pool_factors = pool_factors
        if isinstance(model, nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.activations = dict()
        if pool_factors is None:
            pool_factors = {layer_name: 1 for layer_name in layer_names}

        d = dict(self.model.named_modules())
        # print('Will fetch activations from:')
        for layer_name in layer_names:
            if layer_name in d:
                layer = self.getLayer(layer_name)
                pool_factor = pool_factors[layer_name]
                # layer_rep = re.match('.+($|\n)', layer.__repr__())
                # print('{}, average pooled by {}'.format(layer_name, pool_factor))
                layer.register_forward_hook(self.getActivation(layer_name, pool_factor))
            else:
                print("Warning: Layer {} not found".format(layer_name))

    def __repr__(self):
        out = 'Layers {}\n'.format(self.layer_names)
        if self.pool_factors:
            out = '{}Pool factors {}\n'.format(out, list(self.pool_factors.values()))
        out = '{}'.format(self.model.__repr__())
        return out


    def getActivation(self, name, pool):
        def hook(_, __, output):
            layer_out = output.detach()

            if layer_out.dim() == 4 and pool > 1:
                layer_out_pool = F.avg_pool2d(layer_out, pool)
            elif layer_out.dim() == 4 and pool == -1:
                layer_out_pool = F.avg_pool2d(layer_out, layer_out.size()[-1])
            else:
                layer_out_pool = layer_out
            # print(layer_out_pool.shape)
            if len(layer_out_pool.shape)>2:
                self.activations[name] = layer_out_pool.view(output.size(0), -1)
            else:
                self.activations[name] = layer_out_pool
        return hook

    def __call__(self, data, task_num=None, base_apply=True):
        # self.activations.clear()
        if task_num is not None:
            out = self.model(data)
        else:
            out = self.model(data)
        return out, self.activations

    def getLayer(self, layer_name):
        m = self.model
        sep = '.'
        attrs = layer_name.split(sep)
        for a in attrs:
            try:
                i = int(a)
                m = m[i]
            except ValueError:
                m = m.__getattr__(a)
        return m