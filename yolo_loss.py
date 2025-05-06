import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=2, lambda_coord=5.0, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred, target):
        N = pred.size(0)
        pred = pred.view(N, self.S, self.S, self.B * (5 + self.C))
        target = target.view(N, self.S, self.S, self.B * (5 + self.C))

        loss = 0.0
        for b in range(self.B):
            offset = b * (5 + self.C)

            # Masks where object is present
            obj_mask = target[..., offset + 4] > 0
            noobj_mask = ~obj_mask

            # Box coordinate losses (only where object exists)
            pred_box = pred[..., offset:offset+4][obj_mask]
            target_box = target[..., offset:offset+4][obj_mask]
            loss_xy = ((pred_box[..., 0:2] - target_box[..., 0:2]) ** 2).sum()
            loss_wh = ((pred_box[..., 2:4] - target_box[..., 2:4]) ** 2).sum()

            # Objectness loss
            pred_obj = pred[..., offset + 4]
            target_obj = target[..., offset + 4]
            loss_obj = ((pred_obj - target_obj) ** 2)[obj_mask].sum()
            loss_noobj = ((pred_obj - target_obj) ** 2)[noobj_mask].sum()

            # Classification loss
            pred_cls = pred[..., offset+5:offset+5+self.C][obj_mask]
            target_cls = target[..., offset+5:offset+5+self.C][obj_mask]
            loss_cls = ((pred_cls - target_cls) ** 2).sum()

            loss += (
                self.lambda_coord * (loss_xy + loss_wh)
                + loss_obj
                + self.lambda_noobj * loss_noobj
                + loss_cls
            )

        return loss / N  # average over batch size
