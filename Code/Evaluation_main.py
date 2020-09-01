"""
Usage Instructions:

To evaluate meta-trained models
ex:
python Evaluation_main.py --lr-method 'Meta_Learning' --architect 'FCRN' --target 'TNBC' --eval-meta-train-losses 'BCE' --switchaffine True --num-shots 1 3 5 7 10 --selections 1 2 3 4 5 6 7 8 9 10 --statedictepochs 300 --finetune-lr 0.0001 --finetune-epochs 20 --pretrain-name 'test' --finetune-name 'Test_finetune'

To evaluate supervised-trained models
ex:
python Evaluation_main.py --lr-method 'Supervised_Learning' --architect 'FCRN' --target 'TNBC' --switchaffine True --num-shots 1 3 5 7 10 --selections 1 2 3 4 5 6 7 8 9 10 --state_dict_epochs 50 --finetune-lr 0.0001 --finetune-epochs 20 --pretrainid 'test' --finetune-name 'Test_finetune'
"""


import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code import Evaluation
import torch
import numpy as np

def addEvaluationArgs():
    parser = argparse.ArgumentParser(description="Evaluation Arguments")
    parser.add_argument("--lr-method",type=str,default='',help="Enter Meta_Learning or Supervised_Learning")
    parser.add_argument("--finetune", type=int, default=1)
    parser.add_argument("--testfinetune", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--switchaffine", type=int, default=0)
    parser.add_argument("--targets",type=str,nargs="*",default=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                        help="Combination of B5,B39,TNBC,ssTEM,EM")
    parser.add_argument("--architect",type=str,default='FCRN',help="Enter FCRN or UNet")
    parser.add_argument("--eval-meta-train-losses",type=str,nargs="*",default=['BCE', 'BCE_Entropy', 'BCE_Distillation', 'Combined'],
                        help="Combination of BCE,BCE_Entropy,BCE_Distillation,Combined")
    parser.add_argument("--eval-selections",type=int,nargs="*",default=list(range(1,11)),
                        help="Up to 10 selections")
    parser.add_argument("--meta-lr", type=float, default=0.001,
                        help="Pre-trained meta step size")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Pre-trained learning rate")

    parser.add_argument("--finetune-lr", type=float, default=0.1,
                        help="Finetune learning rate")
    parser.add_argument("--finetune-loss", type=str, default="bce",
                        help="Binary Cross entropy Loss (BCE) function or Weighted BCE (weightedbce)")
    parser.add_argument('--meta-epochs', type=int, default=300)
    parser.add_argument('--innerepochs', type=int, default=20)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--statedictepoch', type=int, default=None,help="Load saved parameters from pre-training epoch #")
    parser.add_argument('--num-shots', type=int,nargs="*",default=[1,3,5,7,10])
    parser.add_argument("--pretrained-name", type=str, default='',
                        help="model name to be finetuned and evaluated")

    parser.add_argument("--finetune-name", type=str, default='',
                        help="finetuned model name")
    return parser



def main():


   parser = addEvaluationArgs()
   args = parser.parse_args()
   print(args)

   meta_params = {'methods': args.metamethods,
                  'hyperparams': {'meta_lr': str(args.meta_lr),
                                  'meta_epochs': str(args.meta_epochs),
                                  'model_lr': str(args.lr),
                                  'inner_epochs': str(args.inner_epochs),
                                  'k-shot': '5',
                                  'optimizer': {'weight_decay': '0.0005',
                                                'momentum': '0.9'}}}

   supervised_params = {'model_lr': str(args.lr),
                        'epochs': '100'}

   batchsize_testset = {'TNBC': 32,
                        'B39': 32,
                        'ssTEM': 32,
                        'EM': 20,
                        'B5': 32}

   evaluation_config = {'targets': args.targets,
                        'selections': args.selections,
                        'k-shot': args.numshots,
                        'batchsize_ftset': 64,
                        'batchsize_testset': batchsize_testset,
                        'ft_lr': args.finetune_lr,
                        'ft_epochs': args.finetune_epochs,
                        'optimizer': {'weight_decay': 0.0005,
                                      'momentum': 0.9, },
                        'Finetune': bool(args.finetune),
                        'Test_Finetuned':bool(args.testfinetune)}


   evaluation = Evaluation.Evaluation(lr_method=args.lr_method, evaluation_config=evaluation_config, meta_params=meta_params,
                                      state_dict_epoch=args.statedictepoch,switch_affine=bool(args.switchaffine),
                                      supervised_params=supervised_params,architecture=args.architect,affine=bool(args.affine),
                                      pretrained_experiment_name_postfix=args.pretrained_name,loss_function=args.finetune_loss,
                                      ft_experiment_name_postfix=args.finetune_name)
   print("Evaluating on Selections: ",args.selections)
   print("LR Method: ", args.lr_method)
   if args.lr_method=='Meta_Learning':
       evaluation.evaluate_meta_learning()
   elif args.lr_method=='Supervised_Learning':
       evaluation.evaluate_transfer_learning()




if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    main()

