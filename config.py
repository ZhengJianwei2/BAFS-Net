#this is our model's config file

import argparse

def getConfig():
    parse = argparse.ArgumentParser()

    #Normal setting 
    parse.add_argument('--multi_gpu',type=bool,default=False)
    parse.add_argument('--num_workers',type=int,default=4)
    parse.add_argument('--dataset',type=str,default='ORSSD')
    parse.add_argument('--data_path',type=str,default='SodDataset/ORSSD/trainset/')
    
    #Training stage parameter settings
    parse.add_argument('--img_size',type=int,default=320)
    parse.add_argument('--batch_size', type=int, default=16)
    parse.add_argument('--epochs', type=int, default=60)
    parse.add_argument('--lr', type=float, default=1e-4)
    parse.add_argument('--optimizer', type=str, default='Adam')
    parse.add_argument('--weight_decay', type=float, default=1e-4)
    parse.add_argument('--scheduler', type=str, default='Reduce', help='Reduce or Step')
    parse.add_argument('--lr_factor', type=float, default=0.1)
    parse.add_argument('--grad_clipping', type=float, default=2, help='Gradient clipping')
    parse.add_argument('--patience', type=int, default=100, help="Scheduler ReduceLROnPlateau's parameter & Early Stopping(+5)")
    parse.add_argument('--model_path', type=str, default='results/')
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--save_map', type=bool, default=None, help='Save prediction map')
    parse.add_argument('--save_map_path', type=str, default='resMap/')
    parse.add_argument('--aug_option', type=int, default=2, help='1=Normal, 2=Augmentation')
    parse.add_argument('--criteria', type=int, default=1, help='w/o weighted 1/0')
    parse.add_argument('--action', type=str, default='train', help='train or test or valid')
    parse.add_argument('--snapshot', type=str, default='none', help='pretrained model')
    parse.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
     
    cfg = parse.parse_args()

    return cfg

if __name__ =='__main__':
    cfg = getConfig()
    cfg = vars(cfg)
    print(cfg)