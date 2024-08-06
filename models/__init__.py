# -- coding: utf-8 --
from importlib import import_module
from utils import networks


class Model:
    """
    Used to invoke different network structures in the models module
    """

    def __init__(self, args):
        print('==> Making model......')

        module = import_module('models.' + args.model.lower())
        self.model = module.make_model(args)

    def __repr__(self):
        return "This is the class that calls the different models in the models module."


class Model_adv:
    """
    Used to invoke different network structures in the models module
    """

    def __init__(self, args):
        print('==> Making model......')

        module = import_module('models.' + args.model.lower())
        
        self.netD_A = networks.define_D(args.output_nc, args.ndf, args.netD,
                                         args.n_layers_D, args.norm, args.init_type, args.init_gain, args.gpuid)
        self.netD_A_edge = networks.define_D(args.output_nc, args.ndf, args.netD,
                                            args.n_layers_D, args.norm, args.init_type, args.init_gain, args.gpuid)
        self.model = module.make_model(args)
    
    def __repr__(self):
        return "This is the class that calls the different models in the models module."
    
    def get_discriminator_A(self):
        return self.netD_A
    
    def get_discriminator_A_edge(self):
        return self.netD_A_edge
    
# if __name__ == '__main__':
#     from option import args
#
#     print(Model(args))
