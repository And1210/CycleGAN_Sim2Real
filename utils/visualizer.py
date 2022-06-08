import numpy as np
import sys
from subprocess import Popen, PIPE
from utils.util import tensor2im
import visdom


class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.

        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            self.create_visdom_connections()


    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        # x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        # y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        x = np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1)
        y = np.array(self.loss_plot_data['Y'])
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_current_validation_metrics(self, epoch, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.

        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1)
        except ConnectionError:
            self.create_visdom_connections()


    # def plot_current_epoch_loss(self, epoch, losses):
    #     """Display the current losses on visdom display: dictionary of error labels and values.
    #
    #     Input params:
    #         epoch: Current epoch.
    #         counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
    #         losses: Training losses stored in the format of (name, float) pairs.
    #     """
    #     if not hasattr(self, 'loss_plot_epoch_data'):
    #         self.loss_plot_epoch_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
    #     self.loss_plot_epoch_data['X'].append(epoch)
    #     self.loss_plot_epoch_data['Y'].append([losses[k] for k in self.loss_plot_epoch_data['legend']])
    #     # x = np.squeeze(np.stack([np.array(self.loss_plot_epoch_data['X'])] * len(self.loss_plot_epoch_data['legend']), 1), axis=1)
    #     # y = np.squeeze(np.array(self.loss_plot_epoch_data['Y']), axis=1)
    #     x = np.stack([np.array(self.loss_plot_epoch_data['X'])] * len(self.loss_plot_epoch_data['legend']), 1)
    #     y = np.array(self.loss_plot_epoch_data['Y'])
    #     try:
    #         self.vis.line(
    #             X=x,
    #             Y=y,
    #             opts={
    #                 'title': self.name + ' loss over epoch',
    #                 'legend': self.loss_plot_epoch_data['legend'],
    #                 'xlabel': 'epoch',
    #                 'ylabel': 'loss'},
    #             win=self.display_id+2)
    #     except ConnectionError:
    #         self.create_visdom_connections()
    #
    #
    # def print_current_losses(self, epoch, max_epochs, iter, max_iters, losses):
    #     """Print current losses on console.
    #
    #     Input params:
    #         epoch: Current epoch.
    #         max_epochs: Maximum number of epochs.
    #         iter: Iteration in epoch.
    #         max_iters: Number of iterations in epoch.
    #         losses: Training losses stored in the format of (name, float) pairs
    #     """
    #     message = '[epoch: {}/{}, iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
    #     for k, v in losses.items():
    #         message += '{0}: {1:.6f} '.format(k, v)
    #
    #     print(message)  # print the message


    def plot_roc_curve(self, fpr, tpr, thresholds):
        """Display the ROC curve.

        Input params:
            fpr: False positive rate (1 - specificity).
            tpr: True positive rate (sensitivity).
            thresholds: Thresholds for the curve.
        """
        try:
            self.vis.line(
                X=fpr,
                Y=tpr,
                opts={
                    'title': 'ROC Curve',
                    'xlabel': '1 - specificity',
                    'ylabel': 'sensitivity',
                    'fillarea': True},
                win=self.display_id+2)
        except ConnectionError:
            self.create_visdom_connections()


    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        # zip the images together so that always the image is followed by label is followed by prediction
        images = images.permute(1,0,2,3)
        images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3]))

        # add a channel dimension to the tensor since the excepted format by visdom is (B,C,H,W)
        images = images[:,None,:,:]

        try:
            self.vis.images(images, win=self.display_id+3, nrow=3)
        except ConnectionError:
            self.create_visdom_connections()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()
        # if self.use_wandb:
        #     self.wandb_run.log(losses)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk
        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open('./checkpoints/loss_log.txt', "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.
        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        # if self.display_id > 0:  # show images in the browser using visdom
        ncols = self.ncols
        if ncols > 0:        # show all the images in one visdom panel
            ncols = min(ncols, len(visuals))
            h, w = next(iter(visuals.values())).shape[:2]
            table_css = """<style>
                    table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                    table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)  # create a table css
            # create a table of images.
            title = self.name
            label_html = ''
            label_html_row = ''
            images = []
            idx = 0
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                label_html_row += '<td>%s</td>' % label
                images.append(image_numpy.transpose([2, 0, 1]))
                idx += 1
                if idx % ncols == 0:
                    label_html += '<tr>%s</tr>' % label_html_row
                    label_html_row = ''
            white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
            while idx % ncols != 0:
                images.append(white_image)
                label_html_row += '<td></td>'
                idx += 1
            if label_html_row != '':
                label_html += '<tr>%s</tr>' % label_html_row
            try:
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            except VisdomExceptionBase:
                self.create_visdom_connections()

        else:     # show each image in a separate visdom panel;
            idx = 1
            try:
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1
            except Exception as e:
                print(e)
                self.create_visdom_connections()

        # if self.use_wandb:
        #     columns = [key for key, _ in visuals.items()]
        #     columns.insert(0,'epoch')
        #     result_table = wandb.Table(columns=columns)
        #     table_row = [epoch]
        #     ims_dict = {}
        #     for label, image in visuals.items():
        #         image_numpy = util.tensor2im(image)
        #         wandb_image = wandb.Image(image_numpy)
        #         table_row.append(wandb_image)
        #         ims_dict[label] = wandb_image
        #     self.wandb_run.log(ims_dict)
        #     if epoch != self.current_epoch:
        #         self.current_epoch = epoch
        #         result_table.add_data(*table_row)
        #         self.wandb_run.log({"Result": result_table})
        #
        #
        # if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
        #     self.saved = True
        #     # save images to the disk
        #     for label, image in visuals.items():
        #         image_numpy = util.tensor2im(image)
        #         img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
        #         util.save_image(image_numpy, img_path)
        #
        #     # update website
        #     webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
        #     for n in range(epoch, 0, -1):
        #         webpage.add_header('epoch [%d]' % n)
        #         ims, txts, links = [], [], []
        #
        #         for label, image_numpy in visuals.items():
        #             image_numpy = util.tensor2im(image)
        #             img_path = 'epoch%.3d_%s.png' % (n, label)
        #             ims.append(img_path)
        #             txts.append(label)
        #             links.append(img_path)
        #         webpage.add_images(ims, txts, links, width=self.win_size)
        #     webpage.save()
