from copy import deepcopy
import torch
import torch.nn as nn

class EMA(nn.Module):
    """
    专门为 3D DLD 训练设计的 EMA 类。
    特点：内部维护独立模型，不干扰主模型训练，使用方便。
    """
    def __init__(self, model, decay=0.995, update_every=10):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.step = 0
        
        # 1. 弱引用原模型 (只读，不改它)
        self.source_model = model
        
        # 2. 深拷贝一个影子模型 (用于验证/推理)
        self.ema_model = deepcopy(model)
        
        # 3. 冻结影子模型 (它只靠公式更新，不靠梯度)
        for p in self.ema_model.parameters():
            p.requires_grad = False
            
    def update(self):
        self.step += 1
        # 只有到了指定的步数才更新
        if self.step % self.update_every != 0:
            return

        with torch.no_grad():
            # 遍历参数: shadow = decay * shadow + (1-decay) * current
            source_dict = dict(self.source_model.named_parameters())
            ema_dict = dict(self.ema_model.named_parameters())
            
            for name, source_param in source_dict.items():
                if name in ema_dict:
                    ema_param = ema_dict[name]
                    ema_param.data.mul_(self.decay).add_(source_param.data, alpha=1 - self.decay)
            
            # 遍历 Buffer (BatchNorm 等): 直接覆盖
            source_buffers = dict(self.source_model.named_buffers())
            ema_buffers = dict(self.ema_model.named_buffers())
            
            for name, source_buffer in source_buffers.items():
                if name in ema_buffers:
                    ema_buffers[name].data.copy_(source_buffer.data)

    # 允许像模型一样调用: output = ema(input)
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)