import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MLP(nn.Module):
    def __init__(self, hidden_size=128, t_eps=0.04, transition_rate=1.75):
        super(MLP, self).__init__()
        self.t_eps = t_eps
        self.transition_rate = transition_rate
        
        # 卷积层用于特征提取
        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3), padding=(1, 1))
        
        # 全连接层用于输出
        self.fc1 = nn.Linear(hidden_size * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y, t):
        # 卷积层特征提取
        y_real = y.real
        y_imag = y.imag
        
        y_real = self.relu(self.conv1(y_real))
        y_real = self.relu(self.conv2(y_real))
        
        y_imag = self.relu(self.conv1(y_imag))
        y_imag = self.relu(self.conv2(y_imag))
        
        # 池化操作
        y_real = torch.mean(y_real, dim=[2, 3])
        y_imag = torch.mean(y_imag, dim=[2, 3])
        
        # 将 t 扩展为和 y_real, y_imag 形状一致
        t_expanded = t.unsqueeze(1).expand_as(y_real)
        
        # 拼接特征
        combined_input = torch.cat([y_real, y_imag, t_expanded], dim=-1)
        
        # 全连接层
        base_pred = self.relu(self.fc1(combined_input))
        base_pred = self.sigmoid(self.fc2(base_pred)).view(-1, 1, 1, 1)  # 输出范围压缩到0~1
        
         # 计算导数
        exp_term = torch.exp(-self.transition_rate * t.view(-1, 1, 1, 1))
        
        # 计算输出值
        eta = base_pred * (1 - exp_term) + exp_term

        return eta.view(-1, 1, 1, 1)
    
    def grad_f(self, x, y, t):
        """
        计算 grad_ln_f,现在直接使用网络输出的log_eta
        """
         # 确保 t 需要梯度
        t = t.clone().detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            f = self(x, y, t).to(x.device)
            grad_f = torch.autograd.grad(
                outputs=f,
                inputs=t,
                grad_outputs=torch.ones_like(f),
                create_graph=True
            )[0]

        # print(f't={t}, f={f.item():,.4f}, grad={grad_ln_f}')

        return grad_f  # 形状：[batch_size]