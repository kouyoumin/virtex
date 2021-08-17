import copy
import functools
from typing import Any, Dict

import torch
from torch import nn

from virtex.models.captioning import CaptioningModel
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.modules.textual_heads import TextualHead
from virtex.modules.visual_backbones import VisualBackbone


class MammoCaptioningModel(CaptioningModel):
    r"""
    A model to perform image captioning (in both forward and backward directions
    independently, only in forward direction). It is composed of a
    :class:`~virtex.modules.visual_backbones.VisualBackbone` and a
    :class:`~virtex.modules.textual_heads.TextualHead` on top of it.

    During training, it maximizes the likelihood of ground truth caption
    conditioned on image features. During inference, it predicts a caption for
    an input image through beam search decoding.

    Parameters
    ----------
    visual: virtex.modules.visual_backbones.VisualBackbone
        A :class:`~virtex.modules.visual_backbones.VisualBackbone` which
        computes visual features from an input image.
    textual: virtex.modules.textual_heads.TextualHead
        A :class:`~virtex.modules.textual_heads.TextualHead` which
        makes final predictions conditioned on visual features.
    sos_index: int, optional (default = 1)
        The index of the end token (``[SOS]``) in vocabulary.
    eos_index: int, optional (default = 2)
        The index of the end token (``[EOS]``) in vocabulary.
    caption_backward: bool, optional (default = False)
        Whether to *also* perform captioning in backward direction. Default is
        ``False`` -- only forward captioning is performed. When ``True``, a
        clone of textual head is created, which does not share weights with
        "forward" model except input and output embeddings.
    decoder: Any, optional (default = None)
        An instance of :class:`~virtex.utils.beam_search.AutoRegressiveBeamSearch`
        or :class:`~virtex.utils.nucleus_sampling.AutoRegressiveNucleusSampling`
        for decoding captions during inference (unused during training).
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        caption_backward: bool = False,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            caption_backward=caption_backward,
            sos_index=sos_index,
            eos_index=eos_index,
            decoder=decoder,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        Given a batch of images and captions, compute log likelihood loss per
        caption token during training. During inference (with images), predict
        a caption through either beam search decoding or nucleus sampling.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            A batch of images and (optionally) ground truth caption tokens.
            Possible set of keys: ``{"image_id", "image", "caption_tokens",
            "noitpac_tokens", "caption_lengths"}``.

        Returns
        -------
        Dict[str, Any]

            A dict with the following structure, containing loss for optimization,
            loss components to log directly to tensorboard, and optionally
            predictions.

            .. code-block::

                {
                    "loss": torch.Tensor,
                    "loss_components": {
                        "captioning_forward": torch.Tensor,
                        "captioning_backward": torch.Tensor, (optional)
                    },
                    "predictions": torch.Tensor
                }
        """

        # shape: (batch_size, channels, height, width)
        l_cc_features = self.visual(batch["l_cc_img"])
        l_mlo_features = self.visual(batch["l_mlo_img"])
        r_cc_features = self.visual(batch["r_cc_img"])
        r_mlo_features = self.visual(batch["r_mlo_img"])
        visual_features = torch.cat((torch.cat((r_cc_features, l_cc_features), dim=3), torch.cat((r_mlo_features, l_mlo_features), dim=3)), dim=2)
        batch_size = visual_features.size(0)

        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"]
            caption_lengths = batch["caption_lengths"]

            # shape: (batch_size, max_caption_length, vocab_size)
            output_logits = self.textual(
                visual_features, caption_tokens, caption_lengths
            )
            loss = self.loss(
                output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                caption_tokens[:, 1:].contiguous().view(-1),
            )
            output_dict: Dict[str, Any] = {
                "loss": loss,
                # Single scalar per batch for logging in training script.
                "loss_components": {"captioning_forward": loss.clone().detach()},
            }
            # Do captioning in backward direction if specified.
            if self.caption_backward:
                backward_caption_tokens = batch["noitpac_tokens"]

                backward_output_logits = self.backward_textual(
                    visual_features, backward_caption_tokens, caption_lengths
                )
                backward_loss = self.loss(
                    backward_output_logits[:, :-1]
                    .contiguous()
                    .view(-1, self.textual.vocab_size),
                    backward_caption_tokens[:, 1:].contiguous().view(-1),
                )
                output_dict["loss"] += backward_loss

                # Single scalar per batch for logging in training script.
                output_dict["loss_components"].update(
                    captioning_backward=backward_loss.clone().detach()
                )

            if not self.training:
                # During validation (while pretraining), get best prediction
                # at every timestep.
                output_dict["predictions"] = torch.argmax(output_logits, dim=-1)
        else:
            if self.decoder is None:
                raise ValueError("Decoder for predicting captions is missing!")

            # During inference, get beam search predictions for forward
            # model. Predictions from forward transformer will be shifted
            # right by one timestep.
            start_predictions = visual_features.new_full(
                (batch_size,), self.sos_index
            ).long()
            # Add image features as a default argument to match callable
            # signature accepted by beam search class (partial captions only).
            decoding_step = functools.partial(self.decoding_step, visual_features)

            predicted_caption, _ = self.decoder.search(
                start_predictions, decoding_step
            )
            output_dict = {"predictions": predicted_caption}

        return output_dict


class MammoForwardCaptioningModel(MammoCaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=False`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=False,
            decoder=decoder,
        )


class MammoBidirectionalCaptioningModel(MammoCaptioningModel):
    r"""
    Convenient extension of :class:`~virtex.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=True`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        sos_index: int = 1,
        eos_index: int = 2,
        decoder: Any = None,
    ):
        super().__init__(
            visual,
            textual,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=True,
            decoder=decoder,
        )


# Convenient handle for our main model.
MammoVirTexModel = MammoBidirectionalCaptioningModel
