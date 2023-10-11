import os.path as osp
import torch
import pickle
import yaml
import re

import smplx

import env

from smplifyx.utils import *
from smplifyx.data_parser import OpenPose
from smplifyx.camera import create_camera
from smplifyx.prior import create_prior
from smplifyx.modified_fit_single_frame import fit_single_frame


def fit_pose2character(keypoints,
                       result_fn: str,
                       gender="neutral",
                       use_hands=True,
                       use_face=True,
                       use_cuda=True,
                       overwrite=False):

    if not overwrite:
        if osp.exists(result_fn):
            with open(result_fn, 'rb') as result_file:
                results = pickle.load(result_file)
            return results, result_fn
    
    # Get recommend config from file
    args = {}
    with open(env.SMPLX_PRE_CONFIG, "r") as config_file:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        args = yaml.load(config_file, loader)
    args['use_hands'] = use_hands
    args['use_cuda'] = use_cuda
    args['gender'] = gender
    
    # ignore mid chest and all joints below hip
    args['joints_to_ign'] = [1, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]

    if use_cuda and not torch.cuda.is_available():
        raise Exception('CUDA is not available, exiting!')

    all_gender = ['neutral', 'male', 'female']
    if gender not in all_gender:
        raise Exception(f'gender must be one of those values: {all_gender}')
    
    dtype = torch.float32
    
    joint_mapper = JointMapper(smpl_to_openpose(use_hands=use_hands, use_face=use_face))

    vpose_model_path = env.VPOSE_MODEL
    smplx_model_path = env.MODELS_FOLDER

    model_params = dict(model_path=smplx_model_path,
                        model_type='smplx',
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=False,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype)
    model_params.update({key: val for key, val in args.items() if key not in model_params})

    body_model = smplx.create(**model_params)

    # Create the camera object
    camera = create_camera(dtype=dtype, **args)

    body_pose_prior = create_prior(prior_type='l2', dtype=dtype, **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type='l2',
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type='l2',
            dtype=dtype,
            **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type='l2',
            use_left_hand=True,
            dtype=dtype,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type='l2',
            use_right_hand=True,
            dtype=dtype,
            **rhand_args)
    
    shape_prior = create_prior(prior_type='l2', dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        camera = camera.to(device=device)
        body_model = body_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')
    
    # A weight for every joint of the model
    # The weights for the joint terms in the optimization
    joint_weights = torch.tensor(np.ones(OpenPose.NUM_BODY_JOINTS + 2 * OpenPose.NUM_HAND_JOINTS * use_hands + 2 * use_hands + use_face * 51, dtype=np.float32), dtype=dtype).to(device=device)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    fitting_params = dict(keypoints=keypoints,
                          body_model=body_model,
                          camera=camera,
                          joint_weights=joint_weights,
                          shape_prior=shape_prior,
                          body_pose_prior=body_pose_prior,
                          use_hands=use_hands,
                          left_hand_prior=left_hand_prior,
                          right_hand_prior=right_hand_prior,
                          use_face=use_face,
                          expr_prior=expr_prior,
                          jaw_prior=jaw_prior,
                          angle_prior=angle_prior,
                          use_cuda=use_cuda,
                          vposer_ckpt=vpose_model_path,
                          result_fn=result_fn,
                          optim_jaw=True,
                          optim_hands=True,
                          optim_expression=True,
                          optim_shape=True)
    fitting_params.update({key: val for key, val in args.items() if key not in fitting_params})
    return fit_single_frame(**fitting_params)