from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, AUROC, Precision, Recall, ROC
from torch import nn
import torchvision.models as models
import torch
from torchsummary import summary
from torch.optim.lr_scheduler import MultiStepLR

####### CREATING CNN MODEL CLASS

def create_model(freeze_layers, train_layers):
    model = models.resnet.resnet50(pretrained=True)
    if freeze_layers:
        num_layers_param = len(list(model.parameters()))
        for i, param in enumerate(model.parameters()):
            # i >= (num_layers_param - train_layers) => requires grad
            if (i < num_layers_param - train_layers):
                param.requires_grad = False
    model.fc = nn.Linear(2048, 1)
    return model


class CNNModelFinetune(LightningModule):
    def __init__(self,
                 learning_rate=1e-3,
                 momentum=0.0,
                 train_layers=10,
                 freeze_layers=True
                ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.train_layers = train_layers
        self.freeze_layers = freeze_layers
        
        # Defining model to use:
        self.model = create_model(self.freeze_layers, self.train_layers)
        
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        self.precision_ = Precision(task="binary")
        self.recall = Recall(task="binary")
        #self.roc = ROC(task="binary")
        
        # Cost function:
        self.criterion = nn.BCEWithLogitsLoss()
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.squeeze()
        acc = self.accuracy(preds, y)
        
        loss = self.criterion(preds, y.float())
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_views = torch.zeros((3, y.shape[0], 1), device=self.device)
        for i in range(3):
            logits_views[i] = self(x[i])
        logits = torch.mean(logits_views, axis=0).squeeze()
        
        preds = logits
        
        m = nn.Sigmoid()
        prob = m(preds)
        
        acc = self.accuracy(prob, y)
        auroc = self.auroc(prob, y)
        precision = self.precision_(prob, y)
        recall = self.recall(prob, y)
        #roc = self.roc(prob, y)
        
        loss = self.criterion(logits, y.float())
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_auroc", auroc, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        
        return loss
    
    
    def test_step(self, batch, batch_idx):
        # just reuse validation_step
        return self.validation_step(batch, batch_idx)
    
    
    def configure_optimizers(self):
        params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.freeze_layers:
            params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
                    
        optimizer = torch.optim.Adam(params_to_update, self.learning_rate)
        lr_scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1),
            'monitor': 'val_loss', 
            'name': 'log_lr'
        }
        
        return [optimizer], [lr_scheduler]
              
    