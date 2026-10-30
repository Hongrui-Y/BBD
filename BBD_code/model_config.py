class model_config:
    def __init__(
             self,
            model_name,
            batch_size,
            epoch_num,
            lr,
            weight_decay,
            description,
            ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.weight_decay = weight_decay
        self.description = description
    def __str__(self):
        return (f"Model Name: {self.model_name}\n"
                f"Batch Size: {self.batch_size}\n"
                f"Epoch Number: {self.epoch_num}\n"
                f"Learning Rate: {self.lr}\n"
                f"Weight Decay: {self.weight_decay}\n"
                f"Description: {self.description}")


