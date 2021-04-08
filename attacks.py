import torch
import copy
from torch.autograd.gradcheck import zero_gradients
from time import time
import operator as op
import pdb
from typing import Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import runutils
from utils import normalize, denormalize
import kornia


def fast_gradient_method(model_fn, x, eps, norm,
                         clip_min=None,
                         clip_max=None,
                         y=None,
                         targeted=False,
                         sanity_checks=False,
                         attack_type="yuv"):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    if norm not in [np.inf, 1, 2, 3]:
        raise ValueError("Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm))
    for e in eps:
        if e < 0:
            raise ValueError("eps must be greater than or equal to 0, got {} instead".format(e))
    if eps[0] == 0 and eps[1] == 0 and eps[2] == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                  clip_min, clip_max))

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x), y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    
    
    ####################################
    if attack_type == "yuv":
        #Denormalize the gradient value 
        grads = torch.div(x.grad, torch.as_tensor([0.229, 0.224, 0.225], device=x.device).view(-1, 1, 1))
        grad_x_yuv_adv = kornia.color.yuv_to_rgb(grads)
        optimal_perturbation_yuv = optimize_linear(grad_x_yuv_adv, eps, norm)
        # Add perturbation to original example to obtain adversarial example
        adv_x = kornia.color.yuv_to_rgb(kornia.color.rgb_to_yuv(x) + optimal_perturbation_yuv)
    elif attack_type == "rgb":
        optimal_perturbation = optimize_linear(x.grad, eps, norm)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation
    #####################################



    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
              "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a noacorrrm constraint.

    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)

    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """
    #list of the indices of the channels
    red_ind = list(range(1, len(grad.size())))
    #small value to avoid zero division
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        # magnitudes of the gradients
        abs_grad = torch.abs(grad)
        #signs of the gradients
        sign = torch.sign(grad)

        ori_shape = [1]*len(grad.size())
        ori_shape[0] = grad.size(0)
        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), dim=1)

        # mask of pixels to be perturbed 
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
#         print("Number of perturbed pixels", max_mask.sum().item())
        num_ties = max_mask
        #number of pixels that will be perturbed
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)

        #perturbation: create a mask that will dsitribute the perturbation across the selected pixels 
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(
            avoid_zero_div,
            torch.sum(grad ** 2, red_ind, keepdim=True)
            )
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        one_mask = (
            (square <= avoid_zero_div).to(torch.float) * opt_pert_norm +
            (square > avoid_zero_div).to(torch.float))
#         assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.sum(dim=red_ind, keepdim=True)
        one_mask = (
            (square <= avoid_zero_div).to(torch.float) * opt_pert_norm +
            (square > avoid_zero_div).to(torch.float))
#         assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                              "currently implemented.")

    # Scale perturbation to be the solution for the norm=eps rather than
    scaled_perturbation = torch.mul(optimal_perturbation, torch.as_tensor(eps, device=optimal_perturbation.device).view(-1, 1, 1))
    return scaled_perturbation


##################################################################################################################
##################################################################################################################
##################################################################################################################
def pgd_attack(model, images, labels, eps_y=0.3, alpha_y=2/255, eps_uv=0.3, alpha_uv=2/255, iters=16, color_space="yuv") :
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).cuda()
        cost.backward()
    
        with torch.no_grad():
            grads = images.grad.sign()
        adv_images_y = images[:, 0:1, :, :] + alpha_y*grads[:, 0:1, :, :]
        eta_y = torch.clamp(adv_images_y - ori_images[:, 0:1, :, :], min=-eps_y, max=eps_y)
        
        adv_images_uv = images[:, 1:, :, :] + alpha_uv*grads[:, 1:, :, :]
        eta_uv = torch.clamp(adv_images_uv - ori_images[:, 1:, :, :], min=-eps_uv, max=eps_uv)

        a =  torch.hstack([eta_y, eta_uv[:, 0:1, :, :], eta_uv[:, 1:2, :, :]])
        images = ori_images + a
        images[:, 0:1, :, :] = torch.clip(images[:, 0:1, :, :], 0, 1)
        images[:, 1:3, :, :] = torch.clip(images[:, 1:3, :, :], -0.5, 0.5)
        images = images.detach()
        
    return images

##################################################################################################################
##################################################################################################################
##################################################################################################################

def fgsm_attack(model, images, labels, eps_y, eps_uv, color_space="yuv"):

    images.requires_grad = True
            
    outputs = model(images)
    
    model.zero_grad()
    loss = nn.CrossEntropyLoss()
    cost = loss(outputs, labels).cuda()
    cost.backward()
    
    with torch.no_grad():
        grads = images.grad.sign()

    attack_images = images + eps_y*grads
    attack_images[:, 1:3, :, :] = images[:, 1:3, :, :] + eps_uv*grads[:, 1:3, :, :]
    
    attack_images[:, 0:1, :, :] = torch.clip(attack_images[:, 0:1, :, :], 0, 1)
    attack_images[:, 1:3, :, :] = torch.clip(attack_images[:, 1:3, :, :], -0.5, 0.5)
    
    return attack_images.detach(), grads

##################################################################################################################
##################################################################################################################
##################################################################################################################

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50, attack_on="yuv"):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()
#     print("Hello")
#     if is_cuda:
#         image = image.cuda()
#         net = net.cuda()
#         print("Using GPU")
#     else:
#         print("Using CPU")
#     print("Elapsed time:", time()-start)

    f_image = net.forward(Variable(image, requires_grad=True)).data.flatten()
    I = torch.argsort(f_image.flatten(), descending=True)
#     print("Elapsed time:", time()-start)
    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = torch.clone(image)
    w = torch.zeros(input_shape).cuda()
    r_tot = torch.zeros(input_shape).cuda()
#     print("Elapsed time:", time()-start)
    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
#     print("Label:", imagenet_labels[label.item()])
    
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label
    target_class = -1
    
    channels = []
    if attack_on.__contains__("y"): channels.append(0)
    if attack_on.__contains__("u"): channels.append(1)
    if attack_on.__contains__("v"): channels.append(2)
    w_mask = torch.zeros_like(w)
    w_mask[:, channels, :, :] = 1. 
#     print(w_mask)
#     print("Elapsed time:", time()-start)
    while k_i == label and loop_i < max_iter:
#         print("Iteration:", loop_i)
        pert = 999999
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = torch.clone(x.grad.data)
        grad_orig = torch.mul(grad_orig, w_mask)

#         grad_orig = x.grad.data

        if loop_i == 0:
            last_min = 0
            for k in range(1, num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = torch.clone(x.grad.data)
                cur_grad = torch.mul(cur_grad, w_mask)
    #             cur_grad = x.grad.data

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data

                pert_k = torch.abs(f_k)/torch.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    last_min = k
                elif k - last_min == 5:
                    break
            target_class = last_min
        else:
            zero_gradients(x)

            fs[0, I[target_class]].backward(retain_graph=True)
            cur_grad = torch.clone(x.grad.data)
            cur_grad = torch.mul(cur_grad, w_mask)

    #             cur_grad = x.grad.data

            # set new w_k and new f_k
            w = cur_grad - grad_orig
            f = (fs[0, I[target_class]] - fs[0, I[0]]).data

            pert = torch.abs(f)/torch.linalg.norm(w.flatten())            
        
        
        
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / torch.linalg.norm(w)
        r_tot = r_tot + r_i
        if is_cuda:
            pert_image = image + (1+overshoot)*r_tot.cuda()
        else:
            pert_image = image + (1+overshoot)*r_tot
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = torch.argmax(fs.data)
#         print("Prediction:", imagenet_labels[k_i.item()])
        loop_i += 1

    r_tot = (1+overshoot)*r_tot
#     print("Elapsed time:", time()-start)
    return r_tot, loop_i, label, k_i, pert_image

##################################################################################################################
##################################################################################################################
##################################################################################################################

"""
Carlini-Wagner attack (http://arxiv.org/abs/1608.04644).
For detailed usage see my repo: https://github.com/kkew3/pytorch-cw2.git

Referential implementation:

- https://github.com/carlini/nn_robust_attacks.git (the original implementation)
- https://github.com/rwightman/pytorch-nips2017-attack-example.git
"""



def _var2numpy(var):
    """
    Make Variable to numpy array. No transposition will be made.

    :param var: Variable instance on whatever device
    :type var: Variable
    :return: the corresponding numpy array
    :rtype: np.ndarray
    """
    return var.data.cpu().numpy()


def atanh(x, eps=1e-6):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def from_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus


class CWL2Adversary(object):
    """
    The L2 attack adversary. To enforce the box constraint, the
    change-of-variable trick using tanh-space is adopted.

    The loss function to optimize:

    .. math::
        \\|\\delta\\|_2^2 + c \\cdot f(x + \\delta)

    where :math:`f` is defined as

    .. math::
        f(x') = \\max\\{0, (\\max_{i \\ne t}{Z(x')_i} - Z(x')_t) \\cdot \\tau + \\kappa\\}

    where :math:`\\tau` is :math:`+1` if the adversary performs targeted attack;
    otherwise it's :math:`-1`.

    Usage::

        attacker = L2Adversary()
        # inputs: a batch of input tensors
        # targets: a batch of attack targets
        # model: the model to attack
        advx = attacker(model, inputs, targets)


    The change-of-variable trick
    ++++++++++++++++++++++++++++

    Let :math:`a` be a proper affine transformation.

    1. Given input :math:`x` in image space, map :math:`x` to "tanh-space" by

    .. math:: \\hat{x} = \\tanh^{-1}(a^{-1}(x))

    2. Optimize an adversarial perturbation :math:`m` without constraint in the
    "tanh-space", yielding an adversarial example :math:`w = \\hat{x} + m`; and

    3. Map :math:`w` back to the same image space as the one where :math:`x`
    resides:

    .. math::
        x' = a(\\tanh(w))

    where :math:`x'` is the adversarial example, and :math:`\\delta = x' - x`
    is the adversarial perturbation.

    Since the composition of affine transformation and hyperbolic tangent is
    strictly monotonic, $\\delta = 0$ if and only if $m = 0$.

    Symbols used in docstring
    +++++++++++++++++++++++++

    - ``B``: the batch size
    - ``C``: the number of channels
    - ``H``: the height
    - ``W``: the width
    - ``M``: the number of classes
    """

    def __init__(self, targeted=True, confidence=0.0, c_range=(1e-3, 1e10),
                 search_steps=5, max_steps=1000, abort_early=True,
                 box=(-1., 1.), optimizer_lr=1e-2, init_rand=False, attack_on='yuv'):
        """
        :param targeted: ``True`` to perform targeted attack in ``self.run``
               method
        :type targeted: bool
        :param confidence: the confidence constant, i.e. the $\\kappa$ in paper
        :type confidence: float
        :param c_range: the search range of the constant :math:`c`; should be a
               tuple of form (lower_bound, upper_bound)
        :type c_range: Tuple[float, float]
        :param search_steps: the number of steps to perform binary search of
               the constant :math:`c` over ``c_range``
        :type search_steps: int
        :param max_steps: the maximum number of optimization steps for each
               constant :math:`c`
        :type max_steps: int
        :param abort_early: ``True`` to abort early in process of searching for
               :math:`c` when the loss virtually stops increasing
        :type abort_early: bool
        :param box: a tuple of lower bound and upper bound of the box
        :type box: Tuple[float, float]
        :param optimizer_lr: the base learning rate of the Adam optimizer used
               over the adversarial perturbation in clipped space
        :type optimizer_lr: float
        :param init_rand: ``True`` to initialize perturbation to small Gaussian;
               False is consistent with the original paper, where the
               perturbation is initialized to zero
        :type init_rand: bool
        :rtype: None

        Why to make ``box`` default to (-1., 1.) rather than (0., 1.)? TL;DR the
        domain of the problem in pytorch is [-1, 1] instead of [0, 1].
        According to Xiang Xu (samxucmu@gmail.com)::

        > The reason is that in pytorch a transformation is applied first
        > before getting the input from the data loader. So image in range [0,1]
        > will subtract some mean and divide by std. The normalized input image
        > will now be in range [-1,1]. For this implementation, clipping is
        > actually performed on the image after normalization, not on the
        > original image.

        Why to ``optimizer_lr`` default to 1e-2? The optimizer used in Carlini's
        code adopts 1e-2. In another pytorch implementation
        (https://github.com/rwightman/pytorch-nips2017-attack-example.git),
        though, the learning rate is set to 5e-4.
        """
        if len(c_range) != 2:
            raise TypeError('c_range ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(c_range))
        if c_range[0] >= c_range[1]:
            raise ValueError('c_range lower bound ({}) is expected to be less '
                             'than c_range upper bound ({})'.format(*c_range))
        if len(box) != 2:
            raise TypeError('box ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(box))
        if box[0] >= box[1]:
            raise ValueError('box lower bound ({}) is expected to be less than '
                             'box upper bound ({})'.format(*box))
        self.targeted = targeted
        self.confidence = float(confidence)
        self.c_range = (float(c_range[0]), float(c_range[1]))
        self.binary_search_steps = search_steps
        self.max_steps = max_steps
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.box = tuple(map(float, box))  # type: Tuple[float, float]
        self.optimizer_lr = optimizer_lr

        # `self.init_rand` is not in Carlini's code, it's an attempt in the
        # referencing pytorch implementation to improve the quality of attacks.
        self.init_rand = init_rand

        # Since the larger the `scale_const` is, the more likely a successful
        # attack can be found, `self.repeat` guarantees at least attempt the
        # largest scale_const once. Moreover, since the optimal criterion is the
        # L2 norm of the attack, and the larger `scale_const` is, the larger
        # the L2 norm is, thus less optimal, the last attempt at the largest
        # `scale_const` won't ruin the optimum ever found.
        self.repeat = (self.binary_search_steps >= 10)
        self.channels = []
        if attack_on.__contains__("y"): self.channels.append(0)
        if attack_on.__contains__("u"): self.channels.append(1)
        if attack_on.__contains__("v"): self.channels.append(2)
        
        
        

    def __call__(self, model, inputs, targets, to_numpy=True):
        """
        Produce adversarial examples for ``inputs``.

        :param model: the model to attack
        :type model: nn.Module
        :param inputs: the original images tensor, of dimension [B x C x H x W].
               ``inputs`` can be on either CPU or GPU, but it will eventually be
               moved to the same device as the one the parameters of ``model``
               reside
        :type inputs: torch.FloatTensor
        :param targets: the original image labels, or the attack targets, of
               dimension [B]. If ``self.targeted`` is ``True``, then ``targets``
               is treated as the attack targets, otherwise the labels.
               ``targets`` can be on either CPU or GPU, but it will eventually
               be moved to the same device as the one the parameters of
               ``model`` reside
        :type targets: torch.LongTensor
        :param to_numpy: True to return an `np.ndarray`, otherwise,
               `torch.FloatTensor`
        :type to_numpy: bool
        :return: the adversarial examples on CPU, of dimension [B x C x H x W]
        """
        # sanity check
        assert isinstance(model, nn.Module)
        assert len(inputs.size()) == 4
        assert len(targets.size()) == 1

        # get a copy of targets in numpy before moving to GPU, used when doing
        # the binary search on `scale_const`
        targets_np = targets.clone().cpu().numpy()  # type: np.ndarray

        # the type annotations here are used only for type hinting and do
        # not indicate the actual type (cuda or cpu); same applies to all codes
        # below
#         inputs = runutils.make_cuda_consistent(model, inputs)[0]  # type: torch.FloatTensor
#         targets = runutils.make_cuda_consistent(model, targets)[0]  # type: torch.FloatTensor
        inputs = inputs.cuda()
        targets = targets.cuda()
        # run the model a little bit to get the `num_classes`
        num_classes = model(Variable(inputs[0][None, :].cuda(), requires_grad=False)).size(1)  # type: int
        batch_size = inputs.size(0)  # type: int

        # `lower_bounds_np`, `upper_bounds_np` and `scale_consts_np` are used
        # for binary search of each `scale_const` in the batch. The element-wise
        # inquality holds: lower_bounds_np < scale_consts_np <= upper_bounds_np
        lower_bounds_np = np.zeros(batch_size)
        upper_bounds_np = np.ones(batch_size) * self.c_range[1]
        scale_consts_np = np.ones(batch_size) * self.c_range[0]

        # Optimal attack to be found.
        # The three "placeholders" are defined as:
        # - `o_best_l2`: the least L2 norms
        # - `o_best_l2_ppred`: the perturbed predictions made by the adversarial
        #    perturbations with the least L2 norms
        # - `o_best_advx`: the underlying adversarial example of
        #   `o_best_l2_ppred`
        o_best_l2 = np.ones(batch_size) * np.inf
        o_best_l2_ppred = -np.ones(batch_size)
        o_best_advx = inputs.clone().cpu().numpy()  # type: np.ndarray

        # convert `inputs` to tanh-space
        inputs_tanh = self._to_tanh_space(inputs)  # type: torch.FloatTensor
        inputs_tanh_var = Variable(inputs_tanh, requires_grad=False)

        # the one-hot encoding of `targets`
        targets_oh = torch.zeros(targets.size() + (num_classes,))  # type: torch.FloatTensor
#         targets_oh = runutils.make_cuda_consistent(model, targets_oh)[0]
        targets_oh = targets_oh.cuda()
        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_oh_var = Variable(targets_oh, requires_grad=False)

        # the perturbation variable to optimize.
        # `pert_tanh` is essentially the adversarial perturbation in tanh-space.
        # In Carlini's code it's denoted as `modifier`
        pert_tanh = torch.zeros(inputs.size())  # type: torch.FloatTensor
        if self.init_rand:
            nn.init.normal(pert_tanh, mean=0, std=1e-3)
#         pert_tanh = runutils.make_cuda_consistent(model, pert_tanh)[0]
        pert_tanh = pert_tanh.cuda()
        pert_tanh_var = Variable(pert_tanh, requires_grad=True)
        #mask for the uv channels
        tanh_mask = torch.zeros_like(pert_tanh_var)
        tanh_mask[:, self.channels, :, :] = 1. 
        
        optimizer = optim.Adam([pert_tanh_var], lr=self.optimizer_lr)
        for sstep in range(self.binary_search_steps):
            if self.repeat and sstep == self.binary_search_steps - 1:
                scale_consts_np = upper_bounds_np
            scale_consts = torch.from_numpy(np.copy(scale_consts_np)).float()  # type: torch.FloatTensor
#             scale_consts = runutils.make_cuda_consistent(model, scale_consts)[0]
            scale_consts = scale_consts.cuda()
            scale_consts_var = Variable(scale_consts, requires_grad=False)
            print('Using scale consts:', list(scale_consts_np))  # FIXME

            # the minimum L2 norms of perturbations found during optimization
            best_l2 = np.ones(batch_size) * np.inf
            # the perturbed predictions corresponding to `best_l2`, to be used
            # in binary search of `scale_const`
            best_l2_ppred = -np.ones(batch_size)
            # previous (summed) batch loss, to be used in early stopping policy
            prev_batch_loss = np.inf  # type: float 
            
            for optim_step in range(self.max_steps):
                batch_loss, pert_norms_np, pert_outputs_np, advxs_np = \
                    self._optimize(model, optimizer, inputs_tanh_var,
                                   pert_tanh_var, targets_oh_var,
                                   scale_consts_var, tanh_mask)
#                 if optim_step % 10 == 0: print( 'batch [{}] loss: {}'.format(optim_step, batch_loss))  # FIXME

                if self.abort_early and not optim_step % (self.max_steps // 10):
                    if batch_loss > prev_batch_loss * (1 - self.ae_tol):
                        break
                    prev_batch_loss = batch_loss

                # update best attack found during optimization
                pert_predictions_np = np.argmax(pert_outputs_np, axis=1)
                comp_pert_predictions_np = np.argmax(
                        self._compensate_confidence(pert_outputs_np,
                                                    targets_np),
                        axis=1)
                for i in range(batch_size):
                    l2 = pert_norms_np[i]
                    cppred = comp_pert_predictions_np[i]
                    ppred = pert_predictions_np[i]
                    tlabel = targets_np[i]
                    ax = advxs_np[i]
                    if self._attack_successful(cppred, tlabel):
                        assert cppred == ppred
                        if l2 < best_l2[i]:
                            best_l2[i] = l2
                            best_l2_ppred[i] = ppred
                        if l2 < o_best_l2[i]:
                            o_best_l2[i] = l2
                            o_best_l2_ppred[i] = ppred
                            o_best_advx[i] = ax

            # binary search of `scale_const`
            for i in range(batch_size):
                tlabel = targets_np[i]
                assert best_l2_ppred[i] == -1 or \
                       self._attack_successful(best_l2_ppred[i], tlabel)
                assert o_best_l2_ppred[i] == -1 or \
                       self._attack_successful(o_best_l2_ppred[i], tlabel)
                if best_l2_ppred[i] != -1:
                    # successful; attempt to lower `scale_const` by halving it
                    if scale_consts_np[i] < upper_bounds_np[i]:
                        upper_bounds_np[i] = scale_consts_np[i]
                    # `upper_bounds_np[i] == c_range[1]` implies no solution
                    # found, i.e. upper_bounds_np[i] has never been updated by
                    # scale_consts_np[i] until
                    # `scale_consts_np[i] > 0.1 * c_range[1]`
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                else:
                    # failure; multiply `scale_const` by ten if no solution
                    # found; otherwise do binary search
                    if scale_consts_np[i] > lower_bounds_np[i]:
                        lower_bounds_np[i] = scale_consts_np[i]
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                    else:
                        scale_consts_np[i] *= 10

        if not to_numpy:
            o_best_advx = torch.from_numpy(o_best_advx).float()
        return o_best_advx

    def _optimize(self, model, optimizer, inputs_tanh_var, pert_tanh_var,
                  targets_oh_var, c_var, tanh_mask):
        """
        Optimize for one step.

        :param model: the model to attack
        :type model: nn.Module
        :param optimizer: the Adam optimizer to optimize ``modifier_var``
        :type optimizer: optim.Adam
        :param inputs_tanh_var: the input images in tanh-space
        :type inputs_tanh_var: Variable
        :param pert_tanh_var: the perturbation to optimize in tanh-space,
               ``pert_tanh_var.requires_grad`` flag must be set to True
        :type pert_tanh_var: Variable
        :param targets_oh_var: the one-hot encoded target tensor (the attack
               targets if self.targeted else image labels)
        :type targets_oh_var: Variable
        :param c_var: the constant :math:`c` for each perturbation of a batch,
               a Variable of FloatTensor of dimension [B]
        :type c_var: Variable
        :return: the batch loss, squared L2-norm of adversarial perturbations
                 (of dimension [B]), the perturbed activations (of dimension
                 [B]), the adversarial examples (of dimension [B x C x H x W])
        """
        # the adversarial examples in the image space
        # of dimension [B x C x H x W]        
#         print(torch.mul(pert_tanh_var, tanh_mask)[0, :, 0:5, 0:5])
        
        advxs_var = self._from_tanh_space(inputs_tanh_var + torch.mul(pert_tanh_var, tanh_mask))  # type: Variable
        
        # the perturbed activation before softmax
        pert_outputs_var = model(advxs_var)  # type: Variable
        
        # the original inputs
        inputs_var = self._from_tanh_space(inputs_tanh_var)  # type: Variable

        perts_norm_var = torch.pow(advxs_var - inputs_var, 2)
        perts_norm_var = torch.sum(perts_norm_var.view(
                perts_norm_var.size(0), -1), 1)

        # In Carlini's code, `target_activ_var` is called `real`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `target_activ_var[i]` is the final activation (right before softmax)
        # of the $t$th class, where $t$ is the attack target or the image label
        #
        # noinspection PyArgumentList
        target_activ_var = torch.sum(targets_oh_var * pert_outputs_var, 1)
        inf = 1e4  # sadly pytorch does not work with np.inf;
                   # 1e4 is also used in Carlini's code
        # In Carlini's code, `maxother_activ_var` is called `other`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `maxother_activ_var[i]` is the maximum final activation of all classes
        # other than class $t$, where $t$ is the attack target or the image
        # label.
        #
        # The assertion here ensures (sufficiently yet not necessarily) the
        # assumption behind the trick to get `maxother_activ_var` holds, that
        # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
        # $o_i$ the $i$th element along axis=1 of `pert_outputs_var`.
        #
        # noinspection PyArgumentList
        
        assert (pert_outputs_var.max(1)[0] >= -inf).all(), 'assumption failed'
        # noinspection PyArgumentList
        maxother_activ_var = torch.max(((1 - targets_oh_var) * pert_outputs_var
                                        - targets_oh_var * inf), 1)[0]

        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        # The result `f_var` should be of dimension [B]
        if self.targeted:
            # if targeted, optimize to make `target_activ_var` larger than
            # `maxother_activ_var` by `self.confidence`
            #
            # noinspection PyArgumentList
            f_var = torch.clamp(maxother_activ_var - target_activ_var
                                + self.confidence, min=0.0)
        else:
            # if not targeted, optimize to make `maxother_activ_var` larger than
            # `target_activ_var` (the ground truth image labels) by
            # `self.confidence`
            #
            # noinspection PyArgumentList
            f_var = torch.clamp(target_activ_var - maxother_activ_var
                                + self.confidence, min=0.0)
        # the total loss of current batch, should be of dimension [1] !!!!
        batch_loss_var = torch.sum(perts_norm_var + c_var * f_var)  # type: Variable

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss_var.backward()
        optimizer.step()

        # Make some records in python/numpy on CPU
        batch_loss = batch_loss_var.item()  # type: float
        pert_norms_np = _var2numpy(perts_norm_var)
        pert_outputs_np = _var2numpy(pert_outputs_var)
        advxs_np = _var2numpy(advxs_var)
        return batch_loss, pert_norms_np, pert_outputs_np, advxs_np

    def _attack_successful(self, prediction, target):
        """
        See whether the underlying attack is successful.

        :param prediction: the prediction of the model on an input
        :type prediction: int
        :param target: either the attack target or the ground-truth image label
        :type target: int
        :return: ``True`` if the attack is successful
        :rtype: bool
        """
        if self.targeted:
            return prediction == target
        else:
            return prediction != target

    # noinspection PyUnresolvedReferences
    def _compensate_confidence(self, outputs, targets):
        """
        Compensate for ``self.confidence`` and returns a new weighted sum
        vector.

        :param outputs: the weighted sum right before the last layer softmax
               normalization, of dimension [B x M]
        :type outputs: np.ndarray
        :param targets: either the attack targets or the real image labels,
               depending on whether or not ``self.targeted``, of dimension [B]
        :type targets: np.ndarray
        :return: the compensated weighted sum of dimension [B x M]
        :rtype: np.ndarray
        """
        outputs_comp = np.copy(outputs)
        rng = np.arange(targets.shape[0])
        if self.targeted:
            # for each image $i$:
            # if targeted, `outputs[i, target_onehot]` should be larger than
            # `max(outputs[i, ~target_onehot])` by `self.confidence`
            outputs_comp[rng, targets] -= self.confidence
        else:
            # for each image $i$:
            # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
            # than `outputs[i, target_onehot]` (the ground truth image labels)
            # by `self.confidence`
            outputs_comp[rng, targets] += self.confidence
        return outputs_comp

    def _to_tanh_space(self, x):
        """
        Convert a batch of tensors to tanh-space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension
        """
        return to_tanh_space(x, self.box)

    def _from_tanh_space(self, x):
        """
        Convert a batch of tensors from tanh-space to input space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension;
                 the returned tensor is on the same device as ``x``
        """
        return from_tanh_space(x, self.box)
 


#############################################
imagenet_labels = {0: 'tench, Tinca tinca',
 1: 'goldfish, Carassius auratus',
 2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 3: 'tiger shark, Galeocerdo cuvieri',
 4: 'hammerhead, hammerhead shark',
 5: 'electric ray, crampfish, numbfish, torpedo',
 6: 'stingray',
 7: 'cock',
 8: 'hen',
 9: 'ostrich, Struthio camelus',
 10: 'brambling, Fringilla montifringilla',
 11: 'goldfinch, Carduelis carduelis',
 12: 'house finch, linnet, Carpodacus mexicanus',
 13: 'junco, snowbird',
 14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
 15: 'robin, American robin, Turdus migratorius',
 16: 'bulbul',
 17: 'jay',
 18: 'magpie',
 19: 'chickadee',
 20: 'water ouzel, dipper',
 21: 'kite',
 22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
 23: 'vulture',
 24: 'great grey owl, great gray owl, Strix nebulosa',
 25: 'European fire salamander, Salamandra salamandra',
 26: 'common newt, Triturus vulgaris',
 27: 'eft',
 28: 'spotted salamander, Ambystoma maculatum',
 29: 'axolotl, mud puppy, Ambystoma mexicanum',
 30: 'bullfrog, Rana catesbeiana',
 31: 'tree frog, tree-frog',
 32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
 33: 'loggerhead, loggerhead turtle, Caretta caretta',
 34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
 35: 'mud turtle',
 36: 'terrapin',
 37: 'box turtle, box tortoise',
 38: 'banded gecko',
 39: 'common iguana, iguana, Iguana iguana',
 40: 'American chameleon, anole, Anolis carolinensis',
 41: 'whiptail, whiptail lizard',
 42: 'agama',
 43: 'frilled lizard, Chlamydosaurus kingi',
 44: 'alligator lizard',
 45: 'Gila monster, Heloderma suspectum',
 46: 'green lizard, Lacerta viridis',
 47: 'African chameleon, Chamaeleo chamaeleon',
 48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
 49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
 50: 'American alligator, Alligator mississipiensis',
 51: 'triceratops',
 52: 'thunder snake, worm snake, Carphophis amoenus',
 53: 'ringneck snake, ring-necked snake, ring snake',
 54: 'hognose snake, puff adder, sand viper',
 55: 'green snake, grass snake',
 56: 'king snake, kingsnake',
 57: 'garter snake, grass snake',
 58: 'water snake',
 59: 'vine snake',
 60: 'night snake, Hypsiglena torquata',
 61: 'boa constrictor, Constrictor constrictor',
 62: 'rock python, rock snake, Python sebae',
 63: 'Indian cobra, Naja naja',
 64: 'green mamba',
 65: 'sea snake',
 66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
 67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
 68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
 69: 'trilobite',
 70: 'harvestman, daddy longlegs, Phalangium opilio',
 71: 'scorpion',
 72: 'black and gold garden spider, Argiope aurantia',
 73: 'barn spider, Araneus cavaticus',
 74: 'garden spider, Aranea diademata',
 75: 'black widow, Latrodectus mactans',
 76: 'tarantula',
 77: 'wolf spider, hunting spider',
 78: 'tick',
 79: 'centipede',
 80: 'black grouse',
 81: 'ptarmigan',
 82: 'ruffed grouse, partridge, Bonasa umbellus',
 83: 'prairie chicken, prairie grouse, prairie fowl',
 84: 'peacock',
 85: 'quail',
 86: 'partridge',
 87: 'African grey, African gray, Psittacus erithacus',
 88: 'macaw',
 89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
 90: 'lorikeet',
 91: 'coucal',
 92: 'bee eater',
 93: 'hornbill',
 94: 'hummingbird',
 95: 'jacamar',
 96: 'toucan',
 97: 'drake',
 98: 'red-breasted merganser, Mergus serrator',
 99: 'goose',
 100: 'black swan, Cygnus atratus',
 101: 'tusker',
 102: 'echidna, spiny anteater, anteater',
 103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
 104: 'wallaby, brush kangaroo',
 105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
 106: 'wombat',
 107: 'jellyfish',
 108: 'sea anemone, anemone',
 109: 'brain coral',
 110: 'flatworm, platyhelminth',
 111: 'nematode, nematode worm, roundworm',
 112: 'conch',
 113: 'snail',
 114: 'slug',
 115: 'sea slug, nudibranch',
 116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
 117: 'chambered nautilus, pearly nautilus, nautilus',
 118: 'Dungeness crab, Cancer magister',
 119: 'rock crab, Cancer irroratus',
 120: 'fiddler crab',
 121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
 122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
 123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
 124: 'crayfish, crawfish, crawdad, crawdaddy',
 125: 'hermit crab',
 126: 'isopod',
 127: 'white stork, Ciconia ciconia',
 128: 'black stork, Ciconia nigra',
 129: 'spoonbill',
 130: 'flamingo',
 131: 'little blue heron, Egretta caerulea',
 132: 'American egret, great white heron, Egretta albus',
 133: 'bittern',
 134: 'crane',
 135: 'limpkin, Aramus pictus',
 136: 'European gallinule, Porphyrio porphyrio',
 137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
 138: 'bustard',
 139: 'ruddy turnstone, Arenaria interpres',
 140: 'red-backed sandpiper, dunlin, Erolia alpina',
 141: 'redshank, Tringa totanus',
 142: 'dowitcher',
 143: 'oystercatcher, oyster catcher',
 144: 'pelican',
 145: 'king penguin, Aptenodytes patagonica',
 146: 'albatross, mollymawk',
 147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
 148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
 149: 'dugong, Dugong dugon',
 150: 'sea lion',
 151: 'Chihuahua',
 152: 'Japanese spaniel',
 153: 'Maltese dog, Maltese terrier, Maltese',
 154: 'Pekinese, Pekingese, Peke',
 155: 'Shih-Tzu',
 156: 'Blenheim spaniel',
 157: 'papillon',
 158: 'toy terrier',
 159: 'Rhodesian ridgeback',
 160: 'Afghan hound, Afghan',
 161: 'basset, basset hound',
 162: 'beagle',
 163: 'bloodhound, sleuthhound',
 164: 'bluetick',
 165: 'black-and-tan coonhound',
 166: 'Walker hound, Walker foxhound',
 167: 'English foxhound',
 168: 'redbone',
 169: 'borzoi, Russian wolfhound',
 170: 'Irish wolfhound',
 171: 'Italian greyhound',
 172: 'whippet',
 173: 'Ibizan hound, Ibizan Podenco',
 174: 'Norwegian elkhound, elkhound',
 175: 'otterhound, otter hound',
 176: 'Saluki, gazelle hound',
 177: 'Scottish deerhound, deerhound',
 178: 'Weimaraner',
 179: 'Staffordshire bullterrier, Staffordshire bull terrier',
 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 181: 'Bedlington terrier',
 182: 'Border terrier',
 183: 'Kerry blue terrier',
 184: 'Irish terrier',
 185: 'Norfolk terrier',
 186: 'Norwich terrier',
 187: 'Yorkshire terrier',
 188: 'wire-haired fox terrier',
 189: 'Lakeland terrier',
 190: 'Sealyham terrier, Sealyham',
 191: 'Airedale, Airedale terrier',
 192: 'cairn, cairn terrier',
 193: 'Australian terrier',
 194: 'Dandie Dinmont, Dandie Dinmont terrier',
 195: 'Boston bull, Boston terrier',
 196: 'miniature schnauzer',
 197: 'giant schnauzer',
 198: 'standard schnauzer',
 199: 'Scotch terrier, Scottish terrier, Scottie',
 200: 'Tibetan terrier, chrysanthemum dog',
 201: 'silky terrier, Sydney silky',
 202: 'soft-coated wheaten terrier',
 203: 'West Highland white terrier',
 204: 'Lhasa, Lhasa apso',
 205: 'flat-coated retriever',
 206: 'curly-coated retriever',
 207: 'golden retriever',
 208: 'Labrador retriever',
 209: 'Chesapeake Bay retriever',
 210: 'German short-haired pointer',
 211: 'vizsla, Hungarian pointer',
 212: 'English setter',
 213: 'Irish setter, red setter',
 214: 'Gordon setter',
 215: 'Brittany spaniel',
 216: 'clumber, clumber spaniel',
 217: 'English springer, English springer spaniel',
 218: 'Welsh springer spaniel',
 219: 'cocker spaniel, English cocker spaniel, cocker',
 220: 'Sussex spaniel',
 221: 'Irish water spaniel',
 222: 'kuvasz',
 223: 'schipperke',
 224: 'groenendael',
 225: 'malinois',
 226: 'briard',
 227: 'kelpie',
 228: 'komondor',
 229: 'Old English sheepdog, bobtail',
 230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
 231: 'collie',
 232: 'Border collie',
 233: 'Bouvier des Flandres, Bouviers des Flandres',
 234: 'Rottweiler',
 235: 'German shepherd, German shepherd dog, German police dog, alsatian',
 236: 'Doberman, Doberman pinscher',
 237: 'miniature pinscher',
 238: 'Greater Swiss Mountain dog',
 239: 'Bernese mountain dog',
 240: 'Appenzeller',
 241: 'EntleBucher',
 242: 'boxer',
 243: 'bull mastiff',
 244: 'Tibetan mastiff',
 245: 'French bulldog',
 246: 'Great Dane',
 247: 'Saint Bernard, St Bernard',
 248: 'Eskimo dog, husky',
 249: 'malamute, malemute, Alaskan malamute',
 250: 'Siberian husky',
 251: 'dalmatian, coach dog, carriage dog',
 252: 'affenpinscher, monkey pinscher, monkey dog',
 253: 'basenji',
 254: 'pug, pug-dog',
 255: 'Leonberg',
 256: 'Newfoundland, Newfoundland dog',
 257: 'Great Pyrenees',
 258: 'Samoyed, Samoyede',
 259: 'Pomeranian',
 260: 'chow, chow chow',
 261: 'keeshond',
 262: 'Brabancon griffon',
 263: 'Pembroke, Pembroke Welsh corgi',
 264: 'Cardigan, Cardigan Welsh corgi',
 265: 'toy poodle',
 266: 'miniature poodle',
 267: 'standard poodle',
 268: 'Mexican hairless',
 269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
 270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
 271: 'red wolf, maned wolf, Canis rufus, Canis niger',
 272: 'coyote, prairie wolf, brush wolf, Canis latrans',
 273: 'dingo, warrigal, warragal, Canis dingo',
 274: 'dhole, Cuon alpinus',
 275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
 276: 'hyena, hyaena',
 277: 'red fox, Vulpes vulpes',
 278: 'kit fox, Vulpes macrotis',
 279: 'Arctic fox, white fox, Alopex lagopus',
 280: 'grey fox, gray fox, Urocyon cinereoargenteus',
 281: 'tabby, tabby cat',
 282: 'tiger cat',
 283: 'Persian cat',
 284: 'Siamese cat, Siamese',
 285: 'Egyptian cat',
 286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
 287: 'lynx, catamount',
 288: 'leopard, Panthera pardus',
 289: 'snow leopard, ounce, Panthera uncia',
 290: 'jaguar, panther, Panthera onca, Felis onca',
 291: 'lion, king of beasts, Panthera leo',
 292: 'tiger, Panthera tigris',
 293: 'cheetah, chetah, Acinonyx jubatus',
 294: 'brown bear, bruin, Ursus arctos',
 295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
 296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
 297: 'sloth bear, Melursus ursinus, Ursus ursinus',
 298: 'mongoose',
 299: 'meerkat, mierkat',
 300: 'tiger beetle',
 301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
 302: 'ground beetle, carabid beetle',
 303: 'long-horned beetle, longicorn, longicorn beetle',
 304: 'leaf beetle, chrysomelid',
 305: 'dung beetle',
 306: 'rhinoceros beetle',
 307: 'weevil',
 308: 'fly',
 309: 'bee',
 310: 'ant, emmet, pismire',
 311: 'grasshopper, hopper',
 312: 'cricket',
 313: 'walking stick, walkingstick, stick insect',
 314: 'cockroach, roach',
 315: 'mantis, mantid',
 316: 'cicada, cicala',
 317: 'leafhopper',
 318: 'lacewing, lacewing fly',
 319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
 320: 'damselfly',
 321: 'admiral',
 322: 'ringlet, ringlet butterfly',
 323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
 324: 'cabbage butterfly',
 325: 'sulphur butterfly, sulfur butterfly',
 326: 'lycaenid, lycaenid butterfly',
 327: 'starfish, sea star',
 328: 'sea urchin',
 329: 'sea cucumber, holothurian',
 330: 'wood rabbit, cottontail, cottontail rabbit',
 331: 'hare',
 332: 'Angora, Angora rabbit',
 333: 'hamster',
 334: 'porcupine, hedgehog',
 335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
 336: 'marmot',
 337: 'beaver',
 338: 'guinea pig, Cavia cobaya',
 339: 'sorrel',
 340: 'zebra',
 341: 'hog, pig, grunter, squealer, Sus scrofa',
 342: 'wild boar, boar, Sus scrofa',
 343: 'warthog',
 344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
 345: 'ox',
 346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
 347: 'bison',
 348: 'ram, tup',
 349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
 350: 'ibex, Capra ibex',
 351: 'hartebeest',
 352: 'impala, Aepyceros melampus',
 353: 'gazelle',
 354: 'Arabian camel, dromedary, Camelus dromedarius',
 355: 'llama',
 356: 'weasel',
 357: 'mink',
 358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
 359: 'black-footed ferret, ferret, Mustela nigripes',
 360: 'otter',
 361: 'skunk, polecat, wood pussy',
 362: 'badger',
 363: 'armadillo',
 364: 'three-toed sloth, ai, Bradypus tridactylus',
 365: 'orangutan, orang, orangutang, Pongo pygmaeus',
 366: 'gorilla, Gorilla gorilla',
 367: 'chimpanzee, chimp, Pan troglodytes',
 368: 'gibbon, Hylobates lar',
 369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
 370: 'guenon, guenon monkey',
 371: 'patas, hussar monkey, Erythrocebus patas',
 372: 'baboon',
 373: 'macaque',
 374: 'langur',
 375: 'colobus, colobus monkey',
 376: 'proboscis monkey, Nasalis larvatus',
 377: 'marmoset',
 378: 'capuchin, ringtail, Cebus capucinus',
 379: 'howler monkey, howler',
 380: 'titi, titi monkey',
 381: 'spider monkey, Ateles geoffroyi',
 382: 'squirrel monkey, Saimiri sciureus',
 383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
 384: 'indri, indris, Indri indri, Indri brevicaudatus',
 385: 'Indian elephant, Elephas maximus',
 386: 'African elephant, Loxodonta africana',
 387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
 388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
 389: 'barracouta, snoek',
 390: 'eel',
 391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
 392: 'rock beauty, Holocanthus tricolor',
 393: 'anemone fish',
 394: 'sturgeon',
 395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
 396: 'lionfish',
 397: 'puffer, pufferfish, blowfish, globefish',
 398: 'abacus',
 399: 'abaya',
 400: "academic gown, academic robe, judge's robe",
 401: 'accordion, piano accordion, squeeze box',
 402: 'acoustic guitar',
 403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
 404: 'airliner',
 405: 'airship, dirigible',
 406: 'altar',
 407: 'ambulance',
 408: 'amphibian, amphibious vehicle',
 409: 'analog clock',
 410: 'apiary, bee house',
 411: 'apron',
 412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
 413: 'assault rifle, assault gun',
 414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
 415: 'bakery, bakeshop, bakehouse',
 416: 'balance beam, beam',
 417: 'balloon',
 418: 'ballpoint, ballpoint pen, ballpen, Biro',
 419: 'Band Aid',
 420: 'banjo',
 421: 'bannister, banister, balustrade, balusters, handrail',
 422: 'barbell',
 423: 'barber chair',
 424: 'barbershop',
 425: 'barn',
 426: 'barometer',
 427: 'barrel, cask',
 428: 'barrow, garden cart, lawn cart, wheelbarrow',
 429: 'baseball',
 430: 'basketball',
 431: 'bassinet',
 432: 'bassoon',
 433: 'bathing cap, swimming cap',
 434: 'bath towel',
 435: 'bathtub, bathing tub, bath, tub',
 436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
 437: 'beacon, lighthouse, beacon light, pharos',
 438: 'beaker',
 439: 'bearskin, busby, shako',
 440: 'beer bottle',
 441: 'beer glass',
 442: 'bell cote, bell cot',
 443: 'bib',
 444: 'bicycle-built-for-two, tandem bicycle, tandem',
 445: 'bikini, two-piece',
 446: 'binder, ring-binder',
 447: 'binoculars, field glasses, opera glasses',
 448: 'birdhouse',
 449: 'boathouse',
 450: 'bobsled, bobsleigh, bob',
 451: 'bolo tie, bolo, bola tie, bola',
 452: 'bonnet, poke bonnet',
 453: 'bookcase',
 454: 'bookshop, bookstore, bookstall',
 455: 'bottlecap',
 456: 'bow',
 457: 'bow tie, bow-tie, bowtie',
 458: 'brass, memorial tablet, plaque',
 459: 'brassiere, bra, bandeau',
 460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
 461: 'breastplate, aegis, egis',
 462: 'broom',
 463: 'bucket, pail',
 464: 'buckle',
 465: 'bulletproof vest',
 466: 'bullet train, bullet',
 467: 'butcher shop, meat market',
 468: 'cab, hack, taxi, taxicab',
 469: 'caldron, cauldron',
 470: 'candle, taper, wax light',
 471: 'cannon',
 472: 'canoe',
 473: 'can opener, tin opener',
 474: 'cardigan',
 475: 'car mirror',
 476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
 477: "carpenter's kit, tool kit",
 478: 'carton',
 479: 'car wheel',
 480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
 481: 'cassette',
 482: 'cassette player',
 483: 'castle',
 484: 'catamaran',
 485: 'CD player',
 486: 'cello, violoncello',
 487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
 488: 'chain',
 489: 'chainlink fence',
 490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
 491: 'chain saw, chainsaw',
 492: 'chest',
 493: 'chiffonier, commode',
 494: 'chime, bell, gong',
 495: 'china cabinet, china closet',
 496: 'Christmas stocking',
 497: 'church, church building',
 498: 'cinema, movie theater, movie theatre, movie house, picture palace',
 499: 'cleaver, meat cleaver, chopper',
 500: 'cliff dwelling',
 501: 'cloak',
 502: 'clog, geta, patten, sabot',
 503: 'cocktail shaker',
 504: 'coffee mug',
 505: 'coffeepot',
 506: 'coil, spiral, volute, whorl, helix',
 507: 'combination lock',
 508: 'computer keyboard, keypad',
 509: 'confectionery, confectionary, candy store',
 510: 'container ship, containership, container vessel',
 511: 'convertible',
 512: 'corkscrew, bottle screw',
 513: 'cornet, horn, trumpet, trump',
 514: 'cowboy boot',
 515: 'cowboy hat, ten-gallon hat',
 516: 'cradle',
 517: 'crane',
 518: 'crash helmet',
 519: 'crate',
 520: 'crib, cot',
 521: 'Crock Pot',
 522: 'croquet ball',
 523: 'crutch',
 524: 'cuirass',
 525: 'dam, dike, dyke',
 526: 'desk',
 527: 'desktop computer',
 528: 'dial telephone, dial phone',
 529: 'diaper, nappy, napkin',
 530: 'digital clock',
 531: 'digital watch',
 532: 'dining table, board',
 533: 'dishrag, dishcloth',
 534: 'dishwasher, dish washer, dishwashing machine',
 535: 'disk brake, disc brake',
 536: 'dock, dockage, docking facility',
 537: 'dogsled, dog sled, dog sleigh',
 538: 'dome',
 539: 'doormat, welcome mat',
 540: 'drilling platform, offshore rig',
 541: 'drum, membranophone, tympan',
 542: 'drumstick',
 543: 'dumbbell',
 544: 'Dutch oven',
 545: 'electric fan, blower',
 546: 'electric guitar',
 547: 'electric locomotive',
 548: 'entertainment center',
 549: 'envelope',
 550: 'espresso maker',
 551: 'face powder',
 552: 'feather boa, boa',
 553: 'file, file cabinet, filing cabinet',
 554: 'fireboat',
 555: 'fire engine, fire truck',
 556: 'fire screen, fireguard',
 557: 'flagpole, flagstaff',
 558: 'flute, transverse flute',
 559: 'folding chair',
 560: 'football helmet',
 561: 'forklift',
 562: 'fountain',
 563: 'fountain pen',
 564: 'four-poster',
 565: 'freight car',
 566: 'French horn, horn',
 567: 'frying pan, frypan, skillet',
 568: 'fur coat',
 569: 'garbage truck, dustcart',
 570: 'gasmask, respirator, gas helmet',
 571: 'gas pump, gasoline pump, petrol pump, island dispenser',
 572: 'goblet',
 573: 'go-kart',
 574: 'golf ball',
 575: 'golfcart, golf cart',
 576: 'gondola',
 577: 'gong, tam-tam',
 578: 'gown',
 579: 'grand piano, grand',
 580: 'greenhouse, nursery, glasshouse',
 581: 'grille, radiator grille',
 582: 'grocery store, grocery, food market, market',
 583: 'guillotine',
 584: 'hair slide',
 585: 'hair spray',
 586: 'half track',
 587: 'hammer',
 588: 'hamper',
 589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
 590: 'hand-held computer, hand-held microcomputer',
 591: 'handkerchief, hankie, hanky, hankey',
 592: 'hard disc, hard disk, fixed disk',
 593: 'harmonica, mouth organ, harp, mouth harp',
 594: 'harp',
 595: 'harvester, reaper',
 596: 'hatchet',
 597: 'holster',
 598: 'home theater, home theatre',
 599: 'honeycomb',
 600: 'hook, claw',
 601: 'hoopskirt, crinoline',
 602: 'horizontal bar, high bar',
 603: 'horse cart, horse-cart',
 604: 'hourglass',
 605: 'iPod',
 606: 'iron, smoothing iron',
 607: "jack-o'-lantern",
 608: 'jean, blue jean, denim',
 609: 'jeep, landrover',
 610: 'jersey, T-shirt, tee shirt',
 611: 'jigsaw puzzle',
 612: 'jinrikisha, ricksha, rickshaw',
 613: 'joystick',
 614: 'kimono',
 615: 'knee pad',
 616: 'knot',
 617: 'lab coat, laboratory coat',
 618: 'ladle',
 619: 'lampshade, lamp shade',
 620: 'laptop, laptop computer',
 621: 'lawn mower, mower',
 622: 'lens cap, lens cover',
 623: 'letter opener, paper knife, paperknife',
 624: 'library',
 625: 'lifeboat',
 626: 'lighter, light, igniter, ignitor',
 627: 'limousine, limo',
 628: 'liner, ocean liner',
 629: 'lipstick, lip rouge',
 630: 'Loafer',
 631: 'lotion',
 632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
 633: "loupe, jeweler's loupe",
 634: 'lumbermill, sawmill',
 635: 'magnetic compass',
 636: 'mailbag, postbag',
 637: 'mailbox, letter box',
 638: 'maillot',
 639: 'maillot, tank suit',
 640: 'manhole cover',
 641: 'maraca',
 642: 'marimba, xylophone',
 643: 'mask',
 644: 'matchstick',
 645: 'maypole',
 646: 'maze, labyrinth',
 647: 'measuring cup',
 648: 'medicine chest, medicine cabinet',
 649: 'megalith, megalithic structure',
 650: 'microphone, mike',
 651: 'microwave, microwave oven',
 652: 'military uniform',
 653: 'milk can',
 654: 'minibus',
 655: 'miniskirt, mini',
 656: 'minivan',
 657: 'missile',
 658: 'mitten',
 659: 'mixing bowl',
 660: 'mobile home, manufactured home',
 661: 'Model T',
 662: 'modem',
 663: 'monastery',
 664: 'monitor',
 665: 'moped',
 666: 'mortar',
 667: 'mortarboard',
 668: 'mosque',
 669: 'mosquito net',
 670: 'motor scooter, scooter',
 671: 'mountain bike, all-terrain bike, off-roader',
 672: 'mountain tent',
 673: 'mouse, computer mouse',
 674: 'mousetrap',
 675: 'moving van',
 676: 'muzzle',
 677: 'nail',
 678: 'neck brace',
 679: 'necklace',
 680: 'nipple',
 681: 'notebook, notebook computer',
 682: 'obelisk',
 683: 'oboe, hautboy, hautbois',
 684: 'ocarina, sweet potato',
 685: 'odometer, hodometer, mileometer, milometer',
 686: 'oil filter',
 687: 'organ, pipe organ',
 688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
 689: 'overskirt',
 690: 'oxcart',
 691: 'oxygen mask',
 692: 'packet',
 693: 'paddle, boat paddle',
 694: 'paddlewheel, paddle wheel',
 695: 'padlock',
 696: 'paintbrush',
 697: "pajama, pyjama, pj's, jammies",
 698: 'palace',
 699: 'panpipe, pandean pipe, syrinx',
 700: 'paper towel',
 701: 'parachute, chute',
 702: 'parallel bars, bars',
 703: 'park bench',
 704: 'parking meter',
 705: 'passenger car, coach, carriage',
 706: 'patio, terrace',
 707: 'pay-phone, pay-station',
 708: 'pedestal, plinth, footstall',
 709: 'pencil box, pencil case',
 710: 'pencil sharpener',
 711: 'perfume, essence',
 712: 'Petri dish',
 713: 'photocopier',
 714: 'pick, plectrum, plectron',
 715: 'pickelhaube',
 716: 'picket fence, paling',
 717: 'pickup, pickup truck',
 718: 'pier',
 719: 'piggy bank, penny bank',
 720: 'pill bottle',
 721: 'pillow',
 722: 'ping-pong ball',
 723: 'pinwheel',
 724: 'pirate, pirate ship',
 725: 'pitcher, ewer',
 726: "plane, carpenter's plane, woodworking plane",
 727: 'planetarium',
 728: 'plastic bag',
 729: 'plate rack',
 730: 'plow, plough',
 731: "plunger, plumber's helper",
 732: 'Polaroid camera, Polaroid Land camera',
 733: 'pole',
 734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
 735: 'poncho',
 736: 'pool table, billiard table, snooker table',
 737: 'pop bottle, soda bottle',
 738: 'pot, flowerpot',
 739: "potter's wheel",
 740: 'power drill',
 741: 'prayer rug, prayer mat',
 742: 'printer',
 743: 'prison, prison house',
 744: 'projectile, missile',
 745: 'projector',
 746: 'puck, hockey puck',
 747: 'punching bag, punch bag, punching ball, punchball',
 748: 'purse',
 749: 'quill, quill pen',
 750: 'quilt, comforter, comfort, puff',
 751: 'racer, race car, racing car',
 752: 'racket, racquet',
 753: 'radiator',
 754: 'radio, wireless',
 755: 'radio telescope, radio reflector',
 756: 'rain barrel',
 757: 'recreational vehicle, RV, R.V.',
 758: 'reel',
 759: 'reflex camera',
 760: 'refrigerator, icebox',
 761: 'remote control, remote',
 762: 'restaurant, eating house, eating place, eatery',
 763: 'revolver, six-gun, six-shooter',
 764: 'rifle',
 765: 'rocking chair, rocker',
 766: 'rotisserie',
 767: 'rubber eraser, rubber, pencil eraser',
 768: 'rugby ball',
 769: 'rule, ruler',
 770: 'running shoe',
 771: 'safe',
 772: 'safety pin',
 773: 'saltshaker, salt shaker',
 774: 'sandal',
 775: 'sarong',
 776: 'sax, saxophone',
 777: 'scabbard',
 778: 'scale, weighing machine',
 779: 'school bus',
 780: 'schooner',
 781: 'scoreboard',
 782: 'screen, CRT screen',
 783: 'screw',
 784: 'screwdriver',
 785: 'seat belt, seatbelt',
 786: 'sewing machine',
 787: 'shield, buckler',
 788: 'shoe shop, shoe-shop, shoe store',
 789: 'shoji',
 790: 'shopping basket',
 791: 'shopping cart',
 792: 'shovel',
 793: 'shower cap',
 794: 'shower curtain',
 795: 'ski',
 796: 'ski mask',
 797: 'sleeping bag',
 798: 'slide rule, slipstick',
 799: 'sliding door',
 800: 'slot, one-armed bandit',
 801: 'snorkel',
 802: 'snowmobile',
 803: 'snowplow, snowplough',
 804: 'soap dispenser',
 805: 'soccer ball',
 806: 'sock',
 807: 'solar dish, solar collector, solar furnace',
 808: 'sombrero',
 809: 'soup bowl',
 810: 'space bar',
 811: 'space heater',
 812: 'space shuttle',
 813: 'spatula',
 814: 'speedboat',
 815: "spider web, spider's web",
 816: 'spindle',
 817: 'sports car, sport car',
 818: 'spotlight, spot',
 819: 'stage',
 820: 'steam locomotive',
 821: 'steel arch bridge',
 822: 'steel drum',
 823: 'stethoscope',
 824: 'stole',
 825: 'stone wall',
 826: 'stopwatch, stop watch',
 827: 'stove',
 828: 'strainer',
 829: 'streetcar, tram, tramcar, trolley, trolley car',
 830: 'stretcher',
 831: 'studio couch, day bed',
 832: 'stupa, tope',
 833: 'submarine, pigboat, sub, U-boat',
 834: 'suit, suit of clothes',
 835: 'sundial',
 836: 'sunglass',
 837: 'sunglasses, dark glasses, shades',
 838: 'sunscreen, sunblock, sun blocker',
 839: 'suspension bridge',
 840: 'swab, swob, mop',
 841: 'sweatshirt',
 842: 'swimming trunks, bathing trunks',
 843: 'swing',
 844: 'switch, electric switch, electrical switch',
 845: 'syringe',
 846: 'table lamp',
 847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
 848: 'tape player',
 849: 'teapot',
 850: 'teddy, teddy bear',
 851: 'television, television system',
 852: 'tennis ball',
 853: 'thatch, thatched roof',
 854: 'theater curtain, theatre curtain',
 855: 'thimble',
 856: 'thresher, thrasher, threshing machine',
 857: 'throne',
 858: 'tile roof',
 859: 'toaster',
 860: 'tobacco shop, tobacconist shop, tobacconist',
 861: 'toilet seat',
 862: 'torch',
 863: 'totem pole',
 864: 'tow truck, tow car, wrecker',
 865: 'toyshop',
 866: 'tractor',
 867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
 868: 'tray',
 869: 'trench coat',
 870: 'tricycle, trike, velocipede',
 871: 'trimaran',
 872: 'tripod',
 873: 'triumphal arch',
 874: 'trolleybus, trolley coach, trackless trolley',
 875: 'trombone',
 876: 'tub, vat',
 877: 'turnstile',
 878: 'typewriter keyboard',
 879: 'umbrella',
 880: 'unicycle, monocycle',
 881: 'upright, upright piano',
 882: 'vacuum, vacuum cleaner',
 883: 'vase',
 884: 'vault',
 885: 'velvet',
 886: 'vending machine',
 887: 'vestment',
 888: 'viaduct',
 889: 'violin, fiddle',
 890: 'volleyball',
 891: 'waffle iron',
 892: 'wall clock',
 893: 'wallet, billfold, notecase, pocketbook',
 894: 'wardrobe, closet, press',
 895: 'warplane, military plane',
 896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
 897: 'washer, automatic washer, washing machine',
 898: 'water bottle',
 899: 'water jug',
 900: 'water tower',
 901: 'whiskey jug',
 902: 'whistle',
 903: 'wig',
 904: 'window screen',
 905: 'window shade',
 906: 'Windsor tie',
 907: 'wine bottle',
 908: 'wing',
 909: 'wok',
 910: 'wooden spoon',
 911: 'wool, woolen, woollen',
 912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
 913: 'wreck',
 914: 'yawl',
 915: 'yurt',
 916: 'web site, website, internet site, site',
 917: 'comic book',
 918: 'crossword puzzle, crossword',
 919: 'street sign',
 920: 'traffic light, traffic signal, stoplight',
 921: 'book jacket, dust cover, dust jacket, dust wrapper',
 922: 'menu',
 923: 'plate',
 924: 'guacamole',
 925: 'consomme',
 926: 'hot pot, hotpot',
 927: 'trifle',
 928: 'ice cream, icecream',
 929: 'ice lolly, lolly, lollipop, popsicle',
 930: 'French loaf',
 931: 'bagel, beigel',
 932: 'pretzel',
 933: 'cheeseburger',
 934: 'hotdog, hot dog, red hot',
 935: 'mashed potato',
 936: 'head cabbage',
 937: 'broccoli',
 938: 'cauliflower',
 939: 'zucchini, courgette',
 940: 'spaghetti squash',
 941: 'acorn squash',
 942: 'butternut squash',
 943: 'cucumber, cuke',
 944: 'artichoke, globe artichoke',
 945: 'bell pepper',
 946: 'cardoon',
 947: 'mushroom',
 948: 'Granny Smith',
 949: 'strawberry',
 950: 'orange',
 951: 'lemon',
 952: 'fig',
 953: 'pineapple, ananas',
 954: 'banana',
 955: 'jackfruit, jak, jack',
 956: 'custard apple',
 957: 'pomegranate',
 958: 'hay',
 959: 'carbonara',
 960: 'chocolate sauce, chocolate syrup',
 961: 'dough',
 962: 'meat loaf, meatloaf',
 963: 'pizza, pizza pie',
 964: 'potpie',
 965: 'burrito',
 966: 'red wine',
 967: 'espresso',
 968: 'cup',
 969: 'eggnog',
 970: 'alp',
 971: 'bubble',
 972: 'cliff, drop, drop-off',
 973: 'coral reef',
 974: 'geyser',
 975: 'lakeside, lakeshore',
 976: 'promontory, headland, head, foreland',
 977: 'sandbar, sand bar',
 978: 'seashore, coast, seacoast, sea-coast',
 979: 'valley, vale',
 980: 'volcano',
 981: 'ballplayer, baseball player',
 982: 'groom, bridegroom',
 983: 'scuba diver',
 984: 'rapeseed',
 985: 'daisy',
 986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
 987: 'corn',
 988: 'acorn',
 989: 'hip, rose hip, rosehip',
 990: 'buckeye, horse chestnut, conker',
 991: 'coral fungus',
 992: 'agaric',
 993: 'gyromitra',
 994: 'stinkhorn, carrion fungus',
 995: 'earthstar',
 996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
 997: 'bolete',
 998: 'ear, spike, capitulum',
 999: 'toilet tissue, toilet paper, bathroom tissue'}
