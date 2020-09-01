"""
Usage Instructions:

To meta train
ex:
python Learning_main.py --lr-method 'Meta_Learning' --architect 'FCRN' --target 'TNBC' --meta-train-losses 'BCE' --metalr 1.0 --lr 0.001 --pretrain-name 'test'

To train using supervised
ex:
python Learning_main.py --lr-method 'Supervised_Learning' --architect 'FCRN' --target 'TNBC' --lr 0.001 --pretrain-name 'test'
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code import Meta_Learning,Supervised_Learning

def addLearningArgs():

    parser = argparse.ArgumentParser(description="Train with a Learning method")
    parser.add_argument('--lr-method', type=str, default='', help="Enter Meta_Learning or Supervised_Learning")
    parser.add_argument("--bceloss", type=str, default="weightedbce",
                        help="standard BCE Loss function or weighted BCE Loss")
    parser.add_argument('--savedir', type=str, default='/Pre-trained/',
                        help="Enter Meta_Learning or Supervised_Learning")
    parser.add_argument('--datasets', type=str, nargs="*", default=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                        help="Combination of B5,B39,TNBC,ssTEM,EM")
    parser.add_argument('--architect', type=str, default='FCRN', help="Enter FCRN or UNet")
    parser.add_argument('--meta-train-losses', type=str, nargs="*",
                        default=['BCE', 'BCE_Entropy', 'BCE_Distillation', 'Combined'],
                        help="Combination of BCE,BCE_Entropy,BCE_Distillation,Combined")
    parser.add_argument('--meta-epochs', type=int, default=300)
    parser.add_argument('--meta-lr', type=float, default=0.001)
    parser.add_argument('--inner-epoch', type=int, default=20, help="# of Inner epochs in meta-training")
    parser.add_argument('--lr', type=float, default=0.01)

    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument('--alpha',type=float,default=0.1)
    parser.add_argument('--beta',type=float,default=0.1)

    parser.add_argument('--epochs', type=int, default=100, help="# of Training epochs in supervised-learning")
    parser.add_argument('--target', type=str, nargs="*", default=['B5', 'B39', 'EM', 'ssTEM', 'TNBC'],
                        help="Define Target dataset in leave-out-one-dataset cross validation")

    parser.add_argument('--name',type=str,default='')

    return parser
def checkmetamethods(metatrainlosses_args,metatrainlosses):
    for metatrainloss in metatrainlosses_args:
        if metatrainloss not in metatrainlosses:
            raise ValueError(metatrainloss+" is undefined")
def main():

    parser = addLearningArgs()
    args = parser.parse_args()
    print(args)
    checkmetamethods(args.metatrainlosses,metatrainlosses=['BCE', 'BCE_Entropy', 'BCE_Distillation', 'Combined'])

    if args.lrmethod=='Meta_Learning':
        hyperparams = {'meta_lr': args.meta_lr,
                       'meta_epochs': args.meta_epochs,
                       'model_lr': args.lr,
                       'inner_epochs': args.inner_epoch,
                       'alpha': args.alpha,
                       'beta': args.beta,
                       'k-shot': 5,
                       'optimizer': {'weight_decay': 0.0005,
                                     'momentum': 0.9}}


        meta_learn = Meta_Learning.Meta_Learning(hyperparams=hyperparams, datasets=args.datasets,target_datasets=args.target,affine=bool(args.affine),
                                                 methods=args.meta_train_losses,architecture=args.architect,loss=args.bceloss,
                                                 experiment_name_postfix=args.name)

        meta_learn.meta_train()


    elif args.lrmethod=='Supervised_Learning':
        hyperparams = {'model_lr': args.lr,
                       'epochs': args.epochs,
                       'batchsize': 64,
                       'optimizer': {'weight_decay': 0.0005,
                                     'momentum': 0.9}
                       }


        datasets_path = '../Datasets/FewShot/Source/'


        supervised_learn = Supervised_Learning.Supervised_Learning(hyperparams=hyperparams, datasets=args.datasets,
                                                                   save_dir=args.savedir,affine=bool(args.affine),
                                                                   datasets_path=datasets_path,loss=args.bceloss,
                                                                   targets=args.target, architecture=args.architect, experiment_name_postfix=args.name)

        supervised_learn.supervised_train()
    else:
        print("Learn Type is undefined")

if __name__ =='__main__':
    main()
