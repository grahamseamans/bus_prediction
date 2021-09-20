import pytorch_lightning 
import torch
from data_types import Data_Info, Model_Params

def get_model(mp: Model_Params, data_info: Data_Info, data):
    class model(pytorch_lightning.core.lightning.LightningModule):
        def __init__(self, data):
            super().__init__()

            self.data = data

            self.embedding_outs = [
                card // mp.embedding_scale + 1 for card in data_info.cardinality
            ]

            self.embeddings = torch.nn.ModuleList(
                [
                    torch.nn.Embedding(card, out_size)
                    for card, out_size in zip(data_info.cardinality, self.embedding_outs)
                ]
            )

            self.total_cat_out = sum(self.embedding_outs)
            self.cat_and_non_cat = self.total_cat_out + len(data_info.non_cat_names)

            self.norm_non_cat = torch.nn.BatchNorm1d(len(data_info.non_cat_names))
            self.norm_1 = torch.nn.BatchNorm1d(mp.conv_channels)

            self.conv_1 = torch.nn.Conv1d(
                self.cat_and_non_cat,
                mp.conv_channels,
                mp.kernel_size,
                padding="same",
            )

            self.conv_2 = torch.nn.Conv1d(
                mp.conv_channels, mp.conv_channels, mp.kernel_size, padding="same"
            )

            self.final = torch.nn.Linear(
                data_info.trip_length * mp.conv_channels, data_info.trip_length
            )

            self.loss = torch.nn.MSELoss()

        def forward(self, x):
            non_category, category, r = x

            x = non_category
            x = x.permute(0, 2, 1)
            normed_non_cats = self.norm_non_cat(x)

            x = category
            x = torch.unbind(x, dim=2)
            x = [embedding(cat) for cat, embedding in zip(x, self.embeddings)]
            for embed in x:
                print(embed.shape)
            dim = 2
            print(f'about to cat these on dim {dim}')
            x = torch.cat(x, dim=dim)
            print('after cat...')
            print(x.shape)
            embedded_cats = x.permute(0, 2, 1)

            print('cats and non cats') 
            print(embedded_cats.shape)
            print(normed_non_cats.shape)
            x = torch.cat([embedded_cats, normed_non_cats], dim=1)
            print('catted...')
            print(x.shape)
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.norm_1(x)
            x = torch.flatten(x, start_dim=1)
            x = self.final(x)
            return x

        def configure_optimizers(self):
            return torch.optim.SGD(
                self.parameters(),
                lr=(mp.learning_rate),
                momentum=0.9,
                weight_decay=0.5,
            )

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            self.log("val_loss", loss)
            return loss

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            x, y = batch
            y_hat = self(x)
            return y_hat
    
    return model(data)

