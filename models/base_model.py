import os
import torch
from utils import get_scheduler
from utils import transfer_to_device
from collections import OrderedDict
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    """

    def __init__(self, configuration):
        """Initialize the BaseModel class.

        Parameters:
            configuration: Configuration dictionary.

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define these lists:
            -- self.network_names (str list):       define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.configuration = configuration
        self.is_train = configuration['is_train']
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        torch.backends.cudnn.benchmark = True
        self.save_dir = configuration['checkpoint_path']
        self.network_names = []
        self.loss_names = []
        self.optimizers = []
        self.visual_names = []


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        The implementation here is just a basic setting of input and label. You may implement
        other functionality in your own model.
        """
        self.input = transfer_to_device(input[0], self.device)
        self.target = transfer_to_device(input[1], self.device)


    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    # @abstractmethod
    # def optimize_parameters(self):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration"""
    #     pass

    def setup(self):
        """Load and print networks; create schedulers.
        """
        if type(self.configuration['load_checkpoint']) == str or self.configuration['load_checkpoint'] >= 0:
            last_checkpoint = self.configuration['load_checkpoint']
        else:
            last_checkpoint = -1

        if type(last_checkpoint) == str or last_checkpoint >= 0:
            # enable restarting training
            self.load_networks(last_checkpoint)
            if self.is_train:
                self.load_optimizers(last_checkpoint)
                for o in self.optimizers:
                    o.param_groups[0]['lr'] = o.param_groups[0]['initial_lr'] # reset learning rate

        self.schedulers = [get_scheduler(optimizer, self.configuration) for optimizer in self.optimizers]

        if type(last_checkpoint) == str:
            for s in self.schedulers:
                for _ in range(self.configuration["scheduler_epoch"]):
                    s.step()
        elif last_checkpoint > 0:
            for s in self.schedulers:
                for _ in range(last_checkpoint):
                    s.step()

        # self.print_networks()

    def train(self):
        """Make models train mode during test time."""
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
                # for n in net:
                #     n.train()

    def eval(self):
        """Make models eval mode during test time."""
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()


    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {0:.7f}'.format(lr))


    def save_networks(self, epoch):
        """Save all the networks to the disk.
        """
        for name in self.network_names:
            if isinstance(name, str):
                save_filename = '{0}_net_{1}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if self.use_cuda:
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)


    def load_networks(self, epoch):
        """Load all the networks from the disk.
        """
        for name in self.network_names:
            if isinstance(name, str):
                load_filename = '{0}_net_{1}.pth'.format(epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                # if isinstance(net, torch.nn.DataParallel):
                    # net = net.module
                print('loading the model from {0}'.format(load_path))
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # print(state_dict.keys())

                net.load_state_dict(state_dict)


    def save_optimizers(self, epoch):
        """Save all the optimizers to the disk for restarting training.
        """
        for i, optimizer in enumerate(self.optimizers):
            save_filename = '{0}_optimizer_{1}.pth'.format(epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)

            torch.save(optimizer.state_dict(), save_path)


    def load_optimizers(self, epoch):
        """Load all the optimizers from the disk.
        """
        for i, optimizer in enumerate(self.optimizers):
            load_filename = '{0}_optimizer_{1}.pth'.format(epoch, i)
            load_path = os.path.join(self.save_dir, load_filename)
            print('loading the optimizer from {0}'.format(load_path))
            state_dict = torch.load(load_path)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            optimizer.load_state_dict(state_dict)


    def print_networks(self):
        """Print the total number of parameters in the network and network architecture.
        """
        print('Networks initialized')
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network {0}] Total number of parameters : {1:.3f} M'.format(name, num_params / 1e6))


    # def set_requires_grad(self, requires_grad=False):
    #     """Set requies_grad for all the networks to avoid unnecessary computations.
    #     """
    #     for name in self.network_names:
    #         if isinstance(name, str):
    #             net = getattr(self, name)
    #             for param in net.parameters():
    #                 param.requires_grad = requires_grad

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret


    def pre_epoch_callback(self, epoch):
        pass


    def post_epoch_callback(self, epoch, visualizer):
        pass


    def get_hyperparam_result(self):
        """Returns the final training result for hyperparameter tuning (e.g. best
            validation loss).
        """
        pass


    def export(self):
        """Exports all the networks of the model using JIT tracing. Requires that the
            input is set.
        """
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, name)
                export_path = os.path.join(self.configuration['export_path'], 'exported_net_{}.pth'.format(name))
                if isinstance(self.input, list): # we have to modify the input for tracing
                    self.input = [tuple(self.input)]
                traced_script_module = torch.jit.trace(net, self.input)
                traced_script_module.save(export_path)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass
