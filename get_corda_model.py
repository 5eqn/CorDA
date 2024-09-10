import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from cordalib.evaluate_utils import evaluate_model
from cordalib.datautils import get_calib_data
from cordalib.act_aware_utils import calib_input_distribution, calib_fisher_info, calib_cov_distribution
from cordalib.decomposition import build_model
import numpy as np
import os

def get_corda_model(model, args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load tokenzier
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # collect data
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.calib_loader_size, seed=args.seed) #256, 128
    
    # collect covariance for CO-SVD or activation for ASVD
    if args.act_aware:
        print('Collect activation-aware data for ASVD ...')
        if "fisher" in args.scaling_method:
            calib_fisher_info(model, calib_loader, args.use_cache)
        if "abs" in args.scaling_method:
            calib_input_distribution(
                model, calib_loader, args.scaling_method, args.use_cache
            )
    elif args.cov_aware:
        print('Collecting covariance data for CovSVD ...')
        calib_cov_distribution(
            model, calib_loader, args.use_cache, args.calib_dataset, args.calib_loader_size, seed=args.seed
        )
    else:
        print('Use the normal SVD ...')

    # perform decomposition
    if args.first_eigen:
        print("\n --- IPA mode: use the first r eigen vecs as adapters --- \n")
    else:
        print("\n --- KPA mode: use the last r eigen vecs as adapters --- \n")
    build_model(model, args)

    # Freeze non-linear layers
    for n, p in model.named_parameters():
        #print(n, p.requires_grad)
        if "ALinear" not in n and "BLinear" not in n and p.requires_grad:
            p.requires_grad = False
            #print("changed as False")
            
    return model
