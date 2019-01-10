import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.training import extensions

# my modules
import cnnsc
import models
import data_builder


class SearchParamsOptuna(data, n_trials=100, epoch=20,
                         batchsize=64, gpu_id=-1):
    """
    Search hyper parameters by optuna
    make instance of optuna trainer.

    >>> optunatrainer = SearchParamsOptuna()
    
    and set N of trials.
    >>> optunatrainer(100)
    """
    def __init__(self) -> None:
        self.data = data
        self.n_trials = n_trials
        self.epoch = epoch
        self.batchsize = batchsize
        self.gpu_id  = gpu_id 
        return

    def __call__(self) -> None:
        # Create a new study.
        self.study = optuna.create_study()
        # Invoke optimization of the objective function.
        self.study.optimize(self.objective, 
                   n_trials=self.n_trials)
        self.df = self.study.trials_dataframe()
        self.get_result()
        return

    def get_result(self):
        print(self.study.best_params)
        print(self.study.best_value)
        print(self.study.best_trial)
        return

    def get_df(self, save=False):
        if save:
            self.df.to_csv("result/optuna_result.csv")
        return self.df

    def objective(self, trial):
        # hyper params
        
        # make classifier instance
        model = L.Classifier(self._build_model(trial),
                             lossfun=F.softmax_cross_entropy, accfun=F.accuracy)
        optimizer = self._build_optimizer(trial, model)

        model.to_gpu(self.gpu_id)

        # Iterator
        random = np.random.RandomState(0)
        train, test = self.data.get_chainer_datasets()
        train_iter = chainer.iterators.SerialIterator(train, self.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, self.batchsize,
                                                     repeat=False, shuffle=False)

        # def updater and Trainer
        updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                                   device=gpu_id)

        trainer = chainer.training.Trainer(updater, (self.epoch, 'epoch'))
        trainer.extend(chainer.training.extensions.Evaluator(test_iter, model,
                                                             device=gpu_id))

        log_report_extension = chainer.training.extensions.LogReport(log_name=None)

        trainer.extend(chainer.training.extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        trainer.extend(log_report_extension)

        # running train
        trainer.run()

        # save result
        log_last = log_report_extension.log[-1]
        for key, value in log_last.items():
            trial.set_user_attr(key, value)

        # return validation error
        val_err = 1.0 - log_report_extension.log[-1]['validation/main/accuracy']
        return val_err

    def _build_model(self, trial):
        model_type = trial.suggest_categorical('model_type',
            ["CNN_rand", "CNN_static", "CNN_non_static", "CNN_multi_ch"])
        filters = trial.suggest_
        # return chainer.Chain instance
        return models.cnn[model_type](embed_weights=self.data.embed_weights,
            conv_filter_windows=trial., n_vocab=data.n_vocab)

    def _build_optimizer(self, trial, model):
        # option of optimizer funciton
        optimizer_name = trial.suggest_categorical('optimizer',
                                                   ['Adam', "AdaDelta" ,'RMSProp'])

        if optimizer_name == 'Adam':
            adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
            optimizer = optimizers.Adam(alpha=adam_alpha)
        elif optimizer_name == "AdaDelta":
            optimizer = optimizers.AdaDelta()
        elif optimizer_name == "RMSprop":
            optimizer = optimizers.RMSprop()

        weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        return optimizer


if __name__ == "__main__":
    pass