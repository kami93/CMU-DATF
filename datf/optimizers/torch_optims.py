import torch.optim as optimizers 
# import pdb; pdb.set_trace()
class adam(optimizers.Adam):
    def __init__(self, **kwargs):
        model = kwargs.get("model_instance", None)
        model_name = kwargs.get("model", None)
        learning_rate = kwargs.get("learning_rate", 0.01)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        return super(adam, self).__init__(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


class sgd(optimizers.SGD):
    def __init__(self, **kwargs):
        model = kwargs.get("model_instance", None)
        model_name = kwargs.get("model", None)
        learning_rate = kwargs.get("learning_rate", 0.01)
        weight_decay = kwargs.get("weight_decay", 1e-4)
        momentum = kwargs.get("momentum", 0.9)
        return super(sgd, self).__init__(model.parameters(), lr=args.learning_rate, momentum = momentum, weight_decay=1e-4)