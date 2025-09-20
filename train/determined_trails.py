from determined.pytorch import PyTorchTrialContext, DataLoader
from torch.nn import functional as F
import determined as det

from train_SAE import SAE_Train_config
from llama3_SAE.modeling_llama3_SAE import TopK, Autoencoder, JumpReLu, HeavyStep
import torch
from datasets import load_from_disk
import json
import logging
from typing import Union, Sequence, Dict
import torch.nn as nn
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext

HF_TOKEN = "YOUR_HF_TOKEN"


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class SAETrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        self.context = context
        # Initialize the trial class and wrap the models, optimizers, and LR schedulers.
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        self.context.logger.info(context.get_hparam("config_path"))
        with open(context.get_hparam("config_path")) as f:
            conf_as_json = json.load(f)

        self.conf = conf = SAE_Train_config(**conf_as_json)
        dataset = load_from_disk(self.conf.ds_path)
        self.label_name = "label" if "RTP" not in self.conf.ds_path else "toxicity"
        self.dm = dataset.with_format("torch")

        if conf.activation == "topk":
            self.context.logger.info(
                f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, k: {conf.k}"
            )
            activation = TopK(conf.k)
        if conf.activation == "topk-sigmoid":
            self.context.logger.info(
                f"TopK-SIGMOID: n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, k: {conf.k}"
            )
            activation = TopK(conf.k, nn.Sigmoid())
        elif conf.activation == "jumprelu":
            self.context.logger.info(
                f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, activation: {conf.activation}, sparsity_coef_warmup_steps: {len(self.dm['train']) * conf.max_epochs * conf.sparstity_coef_warmup_duration}"
            )
            activation = JumpReLu()
        elif conf.activation == "relu":
            self.context.logger.info(
                f"n_inputs: {conf.n_inputs}, n_latents: {conf.n_latents}, activation: {conf.activation}, sparsity_coef_warmup_steps: {len(self.dm['train']) * conf.max_epochs * conf.sparstity_coef_warmup_duration}"
            )
            activation = torch.nn.ReLU()
        else:
            raise (
                NotImplementedError,
                f"Activation '{conf.activation}' not implemented.",
            )

        AE = Autoencoder(
            n_inputs=conf.n_inputs,
            n_latents=conf.n_latents,
            activation=activation,
            tied=True,
            normalize=True,
        )

        if conf.ckpt_path is not None:
            logger.info(f"Loading checkpoint from {conf.ckpt_path}.")
            sae_state_dict = torch.load(conf.ckpt_path)["models_state_dict"][0]
            AE.load_state_dict(sae_state_dict, strict=True)

        self.model = self.context.wrap_model(AE)
        # Initialize the optimizer and wrap it using self.context.wrap_optimizer().
        if self.context.get_hparam("cond_loss_scaling") > 0.0:
            if isinstance(self.model.activation, TopK):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_topk_c
            elif isinstance(self.model.activation, JumpReLu):
                self.heavy_step = HeavyStep()
                self.loss_fn = self.loss_jumprelu_c
            elif isinstance(self.model.activation, torch.nn.ReLU):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_relu_c
        else:
            if isinstance(self.model.activation, TopK):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_topk
            elif isinstance(self.model.activation, JumpReLu):
                self.heavy_step = HeavyStep()
                self.loss_fn = self.loss_jumprelu
            elif isinstance(self.model.activation, torch.nn.ReLU):
                self.heavy_step = self.dummy
                self.loss_fn = self.loss_relu

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.conf.lr)
        )

    def get_sparsity_coef(self):
        if self.step < self.sparstity_coef_warmup:
            self.sparstity_coef += self.sparstity_coef_step
        else:
            self.sparstity_coef = self.max_sparstity_coef

    def dummy(self, **kwargs):
        return torch.tensor(0)

    def cond_loss_fn(self, latents, toxicity):
        return F.binary_cross_entropy(
            F.sigmoid(latents[:, : toxicity.shape[1]]),  
            toxicity,
            reduction="none",
        )  

    def loss_topk_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = normalized_mean_squared_error(recons, acts)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)

        return l2_loss + cond_loss.sum(dim=1), l1_loss, l2_loss, cond_loss, 0.0

    def loss_topk(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = normalized_mean_squared_error(recons, acts)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)

        return l2_loss, l1_loss, l2_loss, cond_loss, 0.0

    def loss_jumprelu_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        heavy_step_loss = self.heavy_step(
            latents, torch.exp(self.model.threshold)
        ).mean(dim=1)

        return (
            l2_loss + cond_loss + self.sparstity_coef * heavy_step_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            heavy_step_loss,
        )

    def loss_jumprelu(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)
        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        heavy_step_loss = self.heavy_step(
            latents, torch.exp(self.model.threshold)
        ).mean(dim=1)

        return (
            l2_loss + self.sparstity_coef * heavy_step_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            heavy_step_loss,
        )

    def loss_relu_c(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)

        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return (
            l2_loss + cond_loss + self.sparstity_coef * l1_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            0.0,
        )

    def loss_relu(self, latents_pre_act, latents, recons, acts, toxicity):
        l1_loss = normalized_L1_loss(latents, acts)
        l2_loss = F.mse_loss(recons, acts, reduction="none").mean(1)

        cond_loss = self.cond_loss_fn(latents_pre_act, toxicity)
        return (
            l2_loss + self.sparstity_coef * l1_loss,
            l1_loss,
            l2_loss,
            cond_loss,
            0.0,
        )

    def calc_sep_loss(self, loss, mask):
        e = 1e-9
        loss_mod_tox = loss * mask
        loss_tox = loss_mod_tox.sum() / ((loss_mod_tox > 0).sum() + e)
        loss_mod_ntox = loss * (mask != True)
        loss_ntox = loss_mod_ntox.sum() / ((loss_mod_ntox > 0).sum() + e)

        return loss_tox, loss_ntox

    def clip_grads(self, params):
        torch.nn.utils.clip_grad_norm_(params, 1.0)

    def grad_scaling(self, grad, tox):
        old_shape = grad.shape
        flat_grad = grad.flatten()
        flat_tox = tox.flatten()
        flat_grad[flat_tox] *= self.context.get_hparam("cond_loss_scaling")
        grad = flat_grad.view(old_shape)
        return grad

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        acts = batch["acts"]  # .type(torch.float16)
        toxicity = batch[self.label_name].float()
        if len(toxicity.shape) < 2:
            logger.warning("Label has wrong dim, adding one with None.")
            toxicity = toxicity[:, None]
        elif len(toxicity.shape) > 2:
            logger.error("Label has wrong dim, dim to large. Abort!")
        else:
            pass

        toxicity_mask = toxicity > 0

        latents_pre_act, latents, recons = self.model(acts)

        loss, l1_loss, l2_loss, cond_loss_pre_rmse, heavy_step_loss = self.loss_fn(
            latents_pre_act, latents, recons, acts, toxicity
        )
        h = cond_loss_pre_rmse.register_hook(
            lambda grad: self.grad_scaling(grad, toxicity_mask)
        )

        loss_mean, l1_loss_mean, l2_loss_mean = (
            loss.mean(),
            l1_loss.mean(),
            l2_loss.mean(),
        )

        loss_mean_t, loss_mean_nt = self.calc_sep_loss(loss, toxicity_mask.any(dim=1))
        l1_loss_mean_t, l1_loss_mean_nt = self.calc_sep_loss(
            l1_loss, toxicity_mask.any(dim=1)
        )
        l2_loss_mean_t, l2_loss_mean_nt = self.calc_sep_loss(
            l2_loss, toxicity_mask.any(dim=1)
        )
        cond_loss_mean_t, cond_loss_mean_nt = self.calc_sep_loss(
            cond_loss_pre_rmse, toxicity_mask
        )

        cond_loss_pre_rmse_mean = cond_loss_pre_rmse.mean()
        heavy_step_loss_mean = (
            heavy_step_loss
            if isinstance(heavy_step_loss, float)
            else heavy_step_loss.mean()
        )

        self.context.backward(loss_mean)
        max_grad = max(
            [
                p.grad.detach().abs().max()
                for p in self.model.parameters()
                if p.grad is not None
            ]
        )
        min_grad = min(
            [
                p.grad.detach().abs().min()
                for p in self.model.parameters()
                if p.grad is not None
            ]
        )
        self.context.step_optimizer(self.optimizer, self.clip_grads)
        h.remove()

        return {
            "loss": loss_mean,
            "loss_t": loss_mean_t,
            "loss_nt": loss_mean_nt,
            "l1_loss": l1_loss_mean,
            "l1_loss_t": l1_loss_mean_t,
            "l1_loss_nt": l1_loss_mean_nt,
            "l2_loss": l2_loss_mean,
            "l2_loss_t": l2_loss_mean_t,
            "l2_loss_nt": l2_loss_mean_nt,
            "heavy step": heavy_step_loss_mean,
            "cond_loss": cond_loss_pre_rmse_mean,
            "cond_loss_t": cond_loss_mean_t,
            "cond_loss_nt": cond_loss_mean_nt,
            "max_grad": max_grad,
            "min_grad": min_grad,
        }

    def evaluate_batch(self, batch: TorchData):
        # Define how to evaluate the model by calculating loss and other metrics
        # for a batch of validation data.
        acts = batch["acts"].cuda()  # .type(torch.float16)
        toxicity = batch[self.label_name].float()
        if len(toxicity.shape) < 2:
            logger.warning("Label has wrong dim, adding one with None.")
            toxicity = toxicity[:, None]
        elif len(toxicity.shape) > 2:
            logger.error("Label has wrong dim, dim to large. Abort!")
        else:
            pass

        toxicity_mask = toxicity > 0

        latents_pre_act, latents, recons = self.model(acts)

        loss, l1_loss, l2_loss, cond_loss_pre_rmse, heavy_step_loss = self.loss_fn(
            latents_pre_act, latents, recons, acts, toxicity
        )
        loss_mean, l1_loss_mean, l2_loss_mean = (
            loss.mean(),
            l1_loss.mean(),
            l2_loss.mean(),
        )
        loss_mean_t, loss_mean_nt = self.calc_sep_loss(loss, toxicity_mask.any(dim=1))
        l1_loss_mean_t, l1_loss_mean_nt = self.calc_sep_loss(
            l1_loss, toxicity_mask.any(dim=1)
        )
        l2_loss_mean_t, l2_loss_mean_nt = self.calc_sep_loss(
            l2_loss, toxicity_mask.any(dim=1)
        )
        cond_loss_mean_t, cond_loss_mean_nt = self.calc_sep_loss(
            cond_loss_pre_rmse, toxicity_mask
        )
        cond_loss_pre_rmse_mean = cond_loss_pre_rmse.mean()
        heavy_step_loss_mean = (
            heavy_step_loss
            if isinstance(heavy_step_loss, float)
            else heavy_step_loss.mean()
        )

        return {
            "val_loss": loss_mean,
            "val_l1_loss": l1_loss_mean,
            "val_l2_loss": l2_loss_mean,
            "val_heavy step": heavy_step_loss_mean,
            "val_cond_loss": cond_loss_pre_rmse_mean,
            "val_loss_t": loss_mean_t,
            "val_l1_loss_t": l1_loss_mean_t,
            "val_l2_loss_t": l2_loss_mean_t,
            "val_cond_loss_t": cond_loss_mean_t,
            "val_loss_nt": loss_mean_nt,
            "val_l1_loss_nt": l1_loss_mean_nt,
            "val_l2_loss_nt": l2_loss_mean_nt,
            "val_cond_loss_nt": cond_loss_mean_nt,
        }

    def build_training_data_loader(self):
        # Create the training data loader.
        # This should return a determined.pytorch.Dataset.
        dataset_train = DataLoader(
            self.dm["train"],
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=64,
            persistent_workers=True,
        )
        return dataset_train

    def build_validation_data_loader(self):
        # Create the validation data loader.
        # This should return a determined.pytorch.Dataset.
        dataset_test = DataLoader(
            self.dm["test"],
            batch_size=self.conf.batch_size, 
            shuffle=False,
            num_workers=64,
            persistent_workers=True,
        )
        return dataset_test


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return ((reconstruction - original_input) ** 2).mean(dim=1) / (
        original_input**2
    ).mean(dim=1)


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)


def main(logger):
    print(det.get_cluster_info())

    # On-cluster: configure the latest checkpoint for pause/resume training functionality.
    latest_checkpoint = det.get_cluster_info().latest_checkpoint
    hparams = det.get_cluster_info().trial.hparams

    with det.pytorch.init(hparams=hparams) as train_context:
        train_context.logger = logger
        trial = SAETrial(train_context)
        trainer = det.pytorch.Trainer(trial, train_context)
        trainer.fit(
            checkpoint_period=det.pytorch.Epoch(10),
            validation_period=det.pytorch.Epoch(5),
            latest_checkpoint=latest_checkpoint,
            checkpoint_policy="best",
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    logger = logging.getLogger(__name__)
    main(logger)
