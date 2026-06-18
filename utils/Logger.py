from time import time


class Logger():
    def __init__(self, args):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.start_time = time()
        self._clean_epoch_stats()

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()

    def log(self, it, iters, losses, samples_right, total_samples):
        for k, v in losses.items():
            self.loss_stats[k] = self.loss_stats.get(k, 0.0) + v * total_samples
        for k, v in samples_right.items():
            self.epoch_stats[k] = self.epoch_stats.get(k, 0.0) + v
        self.total += total_samples

    def end_epoch(self):
        if self.total == 0:
            return
        loss_string = ", ".join(
            ["%s : %.3f" % (k, self.loss_stats[k] / self.total) for k in self.loss_stats])
        acc_string = ", ".join(
            ["%s : %.2f" % (k, 100 * (self.epoch_stats[k] / self.total)) for k in self.epoch_stats])
        print("Epoch %d/%d train - %s - acc %s" % (
            self.current_epoch, self.max_epochs, loss_string, acc_string))

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.loss_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(
            ["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))

    def finish(self):
        print("It took %g" % (time() - self.start_time))
