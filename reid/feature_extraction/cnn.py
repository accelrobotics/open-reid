from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch
import logging
logging.basicConfig(level=logging.INFO)

import cv2
import numpy as np


def extract_cnn_feature(model, inputs, modules=None):

    #logging.info('extract_cnn_feature: inputs: ' + str(inputs.shape))

    im0 = np.array(inputs[0, :])
    im0 = np.swapaxes(im0, 0, 2)
    logging.info("IN: " + str((im0.shape, np.amin(im0), np.amax(im0), im0.dtype)))
    # INFO:root:IN: ((128, 256, 3), -2.117904, 2.64, dtype('float32'))

    # reverse transform to [0, 1] image, by applying mean/stdev in get_data of triplet_loss.py,
    # as per Normalize function: https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor

    #     normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])

    # rev_transf =

    im0[:, :, 0] = im0[:, :, 0] * .229 + 0.485
    im0[:, :, 1] = im0[:, :, 1] * .224 + 0.456
    im0[:, :, 2] = im0[:, :, 2] * .225 + 0.406

    cv2.imshow('im0', im0)
    #cv2.imshow('im0', ((-2.117904 + im0) * 255. / (2.64 + 2.117904)).astype(np.uint8))
    cv2.waitKey(200)

    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()

        out_arr = np.array(outputs[0])
        logging.info("OUT: " + str((out_arr.shape, np.amin(out_arr), np.amax(out_arr), out_arr.dtype)))
        # INFO:root:OUT: ((128,), -0.85427773, 0.9916175, dtype('float32'))
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    ret = list(outputs.values())

    return ret