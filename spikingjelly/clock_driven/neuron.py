from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from . import surrogate, base
import math
try:
    import cupy
    from . import neuron_kernel, cu_kernel_opt
except ImportError:
    neuron_kernel = None


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('spike', 0.)
        else:
            self.register_memory('v', v_reset)
            self.register_memory('spike', 0.)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self, vth):
        self.spike = self.surrogate_function(self.v - vth)

    def neuronal_reset(self, vth):
        if self.detach_reset:
            spike = self.spike.detach()
        else:
            spike = self.spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * vth

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def forward(self, x: torch.Tensor, vth, is_reset):
        self.neuronal_charge(x)
        self.neuronal_fire(vth)
        if is_reset:
            self.neuronal_reset(vth)
        return self.spike


class IFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.,
        v_reset: float = 0.,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False
    ):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class MultiStepIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch'):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'

        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            self.spike_seq, self.v_seq = neuron_kernel.MultiStepIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike_seq = self.spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)


            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False):
        """
        * :ref:`API in English <LIFNode.__init__-en>`

        .. _LIFNode.__init__-cn:

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool


        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})

        * :ref:`中文API <LIFNode.__init__-cn>`

        .. _LIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        .. math::
            V[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset})
        """
        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            self.v = self.v + (x - self.v) / self.tau

        else:
            if isinstance(self.v_reset, float) and self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class QIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau


class EIFNode(BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def neuronal_charge(self, x: torch.Tensor):

        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)

        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp((self.v - self.theta_rh) / self.delta_T)) / self.tau

class MultiStepEIFNode(EIFNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1., v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch'):
        """
        * :ref:`API in English <MultiStepEIFNode.__init__-en>`

        .. _MultiStepEIFNode.__init__-cn:

        ::param tau: 膜电位时间常数
        :type tau: float

        :param delta_T: 陡峭度参数
        :type delta_T: float

        :param theta_rh: 基强度电压阈值
        :type theta_rh: float

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，则电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        多步版本的 :class:`spikingjelly.clock_driven.neuron.EIFNode`。

        .. tip::

        对于多步神经元，输入 ``x_seq.shape = [T, *]``，不仅可以使用 ``.v`` 和 ``.spike`` 获取 ``t = T - 1`` 时刻的电压和脉冲，还能够
        使用 ``.v_seq`` 和 ``.spike_seq`` 获取完整的 ``T`` 个时刻的电压和脉冲。

        .. tip::

            阅读 :doc:`传播模式 <./clock_driven/10_propagation_pattern>` 以获取更多关于单步和多步传播的信息。

        * :ref:`中文API <MultiStepEIFNode.__init__-cn>`

        .. _MultiStepEIFNode.__init__-en:

        :param tau: membrane time constant
        :type tau: float

        :param delta_T: sharpness parameter
        :type delta_T: float

        :param theta_rh: rheobase threshold
        :type theta_rh: float

        :param v_threshold: threshold voltage of neurons
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of neurons. If not ``None``, voltage of neurons that just fired spikes will be set to
            ``v_reset``. If ``None``, voltage of neurons that just fired spikes will subtract ``v_threshold``
        :type v_reset: float

        :param surrogate_function: surrogate function for replacing gradient of spiking functions during back-propagation
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset
        :type detach_reset: bool

        :param backend: use which backend, ``'torch'`` or ``'cupy'``. ``'cupy'`` is faster but only supports GPU
        :type backend: str

        .. admonition:: Tip
            :class: tip

            The input for multi-step neurons are ``x_seq.shape = [T, *]``. We can get membrane potential and spike at
            time-step ``t = T - 1`` by ``.v`` and ``.spike``. We can also get membrane potential and spike at all ``T``
            time-steps by ``.v_seq`` and ``.spike_seq``.

        .. admonition:: Tip
            :class: tip

            Read :doc:`Propagation Pattern <./clock_driven_en/10_propagation_pattern>` for more details about single-step
            and multi-step propagation.
        """
        super().__init__(tau, delta_T, theta_rh, v_threshold, v_rest, v_reset,
                 surrogate_function, detach_reset)
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

        assert backend == 'torch' or backend == 'cupy'
        assert not (backend == 'cupy' and neuron_kernel is None), 'cupy is not installed'
        self.backend = backend

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        self.v_seq = torch.zeros_like(x_seq.data)
        self.spike_seq = torch.zeros_like(x_seq.data)

        if self.backend == 'torch':
            for t in range(x_seq.shape[0]):
                self.spike_seq[t] = super().forward(x_seq[t])
                self.v_seq[t] = self.v
            return self.spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)


            self.spike_seq, self.v_seq = neuron_kernel.MultiStepEIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest, self.theta_rh, self.delta_T, self.detach_reset, self.surrogate_function.cuda_code)

            self.spike_seq = self.spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.spike = self.spike_seq[-1].clone()
            self.v = self.v_seq[-1].clone()

            return self.spike_seq
        else:
            raise NotImplementedError

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'
