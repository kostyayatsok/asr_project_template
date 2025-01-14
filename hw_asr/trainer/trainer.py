import random
from random import shuffle

import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            data_loader,
            text_encoder,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 100

        self.train_metrics = MetricTracker(
            "Metrics/loss", "Metrics/grad_norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "Metrics/loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        if self.writer is not None:
            self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if self.writer is not None:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, mode="train")
                self.writer.add_scalar("Metrics/loss1", batch["loss"].item())

            self.train_metrics.update("Metrics/grad_norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0 and self.writer is not None:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_predictions(part="train", **batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch["audio"])
                self._log_scalars(self.train_metrics)
            if batch_idx >= self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(batch["spectrogram_length"])
        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        metrics.update("Metrics/loss", batch["loss"].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )
            if self.writer is not None:
                self.writer.set_step(epoch * self.len_epoch, "valid")
                self._log_scalars(self.valid_metrics)
                self._log_predictions(part="val", **batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_audio(batch["audio"])

        if self.writer is not None:
            # add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(f"Weighths/{name}", p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            log_probs_length,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        argmax_inds = log_probs.cpu().argmax(-1)
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length)
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds.numpy()) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds.numpy()) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, argmax_texts_raw))
        shuffle(tuples)
        to_log_pred = []
        to_log_pred_raw = []
        for pred, target, raw_pred in tuples[:examples_to_log]:
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            to_log_pred.append(
                f"true: '{target}' | pred: '{pred}' "
                f"| wer: {wer:.2f} | cer: {cer:.2f}\n"
            )
            to_log_pred_raw.append(f"true: '{target}' | pred: '{raw_pred}'\n")
        self.writer.add_text(f"Media/predictions", "< < < < > > > >".join(to_log_pred))
        self.writer.add_text(
            f"Media/predictions_raw", "< < < <> > > >".join(to_log_pred_raw)
        )
        
        beamsearch_texts = [self.text_encoder.ctc_beam_search(log_prob, log_prob_len)[0][0] for log_prob, log_prob_len in zip(log_probs.cpu().numpy(), log_probs_length)]
        tuples = list(zip(beamsearch_texts, text))
        shuffle(tuples)
        to_log_pred = []
        for pred, target in tuples[:examples_to_log]:
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            to_log_pred.append(
                f"true: '{target}' | pred: '{pred}' "
                f"| wer: {wer:.2f} | cer: {cer:.2f}\n"
            )
        self.writer.add_text(f"Media/beamsearch", "< < < < > > > >".join(to_log_pred))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch)
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu()))
        self.writer.add_image("Media/spectrogram", ToTensor()(image))

    def _log_audio(self, audio_batch):
        audio = random.choice(audio_batch)
        self.writer.add_audio("Media/audio", audio)


    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
