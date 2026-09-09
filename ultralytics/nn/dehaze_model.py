"""DehazeDetectionModel: YOLO26 + auxiliary image-reconstruction branch."""
from __future__ import annotations
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.dehaze import Decoder, TransposeDecoder


class DehazeDetectionModel(DetectionModel):
    """YOLO26 detection model with an auxiliary dehazing decoder branch."""

    def __init__(self, cfg="yolo26-dehaze.yaml", ch=3, nc=None, verbose=True):
        # NOTE: dehaze_idx / _dehaze_out must be initialised BEFORE super().__init__()
        # because the parent constructor triggers _predict_once internally (stride
        # probing), which would raise AttributeError otherwise.
        self.dehaze_idx = -1       # sentinel: "not found yet"
        self._dehaze_out = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Now self.model is fully built — locate the single restoration layer.
        # Both Decoder (Converse2D) and TransposeDecoder (matched transposed-conv
        # control, see reviewer request) are valid restoration branches.
        idxs = [i for i, m in enumerate(self.model)
                if isinstance(m, (Decoder, TransposeDecoder))]
        assert len(idxs) == 1, f"Expected exactly one restoration layer, got {len(idxs)}"
        self.dehaze_idx = idxs[0]

    @property
    def loss_names(self):
        """Always 4 loss columns: box, cls, dfl, dehaze.

        Overriding this ensures that both the trainer's tloss accumulator and the
        validator's self.loss are consistently initialised to size 4, preventing
        the shape-mismatch RuntimeError that occurs when loss() returns a different
        number of items during training vs. validation.
        """
        return "box", "cls", "dfl", "dehaze"

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        self._dehaze_out = None
        # dehaze_idx may be the sentinel -1 during the parent __init__ stride-probe
        # forward pass; in that case we simply skip the capture.
        dehaze_idx = getattr(self, "dehaze_idx", -1)
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if i == dehaze_idx:
                self._dehaze_out = x
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        det_loss, det_items = self.criterion(preds, batch)

        # ----- dehazing auxiliary loss -----
        # IMPORTANT: det_items must ALWAYS be extended to 4 elements so that
        # the shape is consistent between training and validation steps.
        # Both trainer.tloss and validator.self.loss are sized from loss_names
        # (4 entries), so every call to loss() must return exactly 4 items.
        if self.training and self._dehaze_out is not None and "clear_img" in batch:
            target = batch["clear_img"].to(self._dehaze_out.device)
            target = target * 2.0 - 1.0       # Decoder 输出在 (-1, 1)
            aux = nn.functional.l1_loss(self._dehaze_out, target)
            lam = getattr(self.args, "dehaze_weight", 0.1)
            det_loss = det_loss + lam * aux
            det_items = torch.cat([det_items, aux.detach().unsqueeze(0)])
        else:
            # Validation / no clear_img: pad with 0.0 to keep shape = (4,)
            det_items = torch.cat([det_items, det_items.new_zeros(1)])

        return det_loss, det_items