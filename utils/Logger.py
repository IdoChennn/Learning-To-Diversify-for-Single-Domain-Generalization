from time import time


class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        self.current_iter = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))

    def finish(self):
        print("It took %g" % (time() - self.start_time))
