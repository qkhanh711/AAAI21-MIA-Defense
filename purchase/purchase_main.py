from purchase_models import *
from purchase_distillation import *
from purchase_advtune import *

'''
check https://github.com/BayesWatch/pytorch-moonshine/blob/master/main.py for distillation loss function
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='distillation experiments')

    parser.add_argument('--debug-', type=str, default='MEDIUM', help='debug level (default: MEDIUM)')
    parser.add_argument('--defence', type=str, default='distillation', help='MIA mitigation type (default: distillation)')
    
    parser.add_argument('--train-len', type=int, default=20000, help='Training data len (default: 20000)')
    parser.add_argument('--ref-len', type=int, default=20000, help='reference/distillation data len (default: 20000)')
    
    
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0005, help='normal learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--tr-epochs', type=int, default=100, help='# of training epochs for adv_regularization | # of training epochs for non-private training for distillation (default: 100)')
    parser.add_argument('--schedule', type=list, default=[30,80], help='training schedule (default: 30,80)')


    parser.add_argument('--distil-batch-size', type=int, default=64, help='batch size (default: 32)')
    parser.add_argument('--distil-lr', type=float, default=0.0005, help='normal learning rate (default: 0.1)')
    parser.add_argument('--distil-gamma', type=float, default=0.5, help='learning rate decay factor (default: 0.5)')
    parser.add_argument('--distil-epochs', type=int, default=200, help='# distillation training epochs (default: 100)')
    parser.add_argument('--distil-schedule', type=list, default=[50,90,150], help='distillation training schedule (default: 50,90,150)')
    parser.add_argument('--distil-temp', type=float, default=1.0, help='temperature of softmax (default: 1.0)')

    
    parser.add_argument('--alpha', type=int, default=0, help='adversarial tuning factor (default: 0)')
    parser.add_argument('--advtune-batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--advtune-lr', type=float, default=0.001, help='normal learning rate (default: 0.001)')
    parser.add_argument('--advtune-gamma', type=float, default=0.1, help='learning rate decay factor (default: 0.1)')
    parser.add_argument('--advtune-tr-epochs', type=int, default=100, help='# of training epochs for adv_regularization | # of training epochs for non-private training for distillation (default: 100)')
    parser.add_argument('--advtune-schedule', type=list, default=[25,80], help='training schedule (default: 25,80)')
    

    parser.add_argument('--train', type=int, default=1, help='train defense (default: 1)')
    parser.add_argument('--evaluate', type=int, default=1, help='evaluate defense (default: 1)')
    parser.add_argument('--attack-epochs', type=int, default=200, help='current epoch number (default: 200)')
    parser.add_argument('--n-classes', type=int, default=100, help='Number of classes (default: 100)')
    
    args = parser.parse_args()
    print(args, "\n \n")    
    try:
        # args.distil_epochs=''.join(args.distil_epochs)
        args.distil_epochs=args.distil_epochs.split(',')
        print('Processing distillation defence', args.distil_epochs, type(args.distil_epochs))
    except:
        pass
    
    try:
        if isinstance(args.distil_epochs, str):
            args.distil_epochs = args.distil_epochs.split(',')
            args.distil_epochs=list(map(int,args.distil_epochs))
            print('Processing distillation defence', args.distil_epochs, type(args.distil_epochs))
    except:
        pass
    assert torch.cuda.is_available(), 'No GPUs available... Exiting!'
    
    use_cuda=True

    if args.defence=='distillation':
        print(args, "\n \n")
        print('Processing distillation defence')
        distillation_defense(args.train,args.evaluate)
    
    elif args.defence=='advtune':
        print('Processing adversarial regularization defence')
        advtune_defense(args.train,args.evaluate)
    
    else:
        assert False, 'Unknown defence... Exiting!'

if __name__ == '__main__':
    main()