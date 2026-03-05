
Optim
#####

VMCOptimizer
========================

``vmc/optim/optimizer/VMCOptimizer``

.. code-block:: python
    :linenos:

    class VMCOptimizer(BaseVMCOptimizer):

        def __init__(
            self,
            nqs: DDP,
            sampler_param: dict,
            electron_info: ElectronInfo,
            opt: Optimizer,
            lr_scheduler: Union[List[LRScheduler], LRScheduler] = None,
            max_iter: int = 2000,
            dtype: Dtype = None,
            HF_init: int = 0,
            external_model: any = None,
            check_point: str = None,
            read_model_only: bool = False,
            only_sample: bool = False,
            pre_CI: CIWavefunction = None,
            pre_train_info: dict = None,
            clean_opt_state: bool = False,
            noise_lambda: float = 0.05,
            method_grad: str = "AD",
            sr: bool = False,
            method_jacobian: str = "vector",
            interval: int = 100,
            prefix: str = "VMC",
            MAX_AD_DIM: int = -1,
            kfac: KFACPreconditioner = None,  # type: ignore
            use_clip_grad: bool = False,
            max_grad_norm: float = 1.0,
            max_grad_value: float = 1.0,
            start_clip_grad: int = None,
            clip_grad_method: str = "l2",
            clip_grad_scheduler: Optional[Callable[[int], float]] = None,
            use_3sigma: bool = False,
            k_step_clip: int = 100,
            use_spin_raising: bool = False,
            spin_raising_coeff: float = 1.0,
            only_output_spin_raising: bool = False,
            spin_raising_scheduler: Optional[Callable[[int], float]] = None,
        )

.. _opt-params:

----------
opt-params
----------

.. code-block:: python
    :linenos:


    from utils import ElectronInfo, Dtype

    opt_type = optim.AdamW
    opt_params = {"lr": 0.001, "betas": (0.9, 0.999)}
    opt = opt_type(model.parameters(), **opt_params)

    prefix = "vmc"
    def clip_grad_scheduler(step):
       if step <= 4000:
          max_grad = 1.0
       elif step <= 8000:
          max_grad = 0.1 
       else:
          max_grad = 0.01
       return max_grad

    vmc_opt_params = {
        "nqs": model, 
        "opt": opt,
        # "lr_scheduler": lr_scheduler,
        # "read_model_only": True,
        "dtype": dtype,
        "sampler_param": sampler_param,
        # "only_sample": True,
        "electron_info": electron_info,
        # "use_spin_raising": True,
        # "spin_raising_coeff": 1.0,
        # "only_output_spin_raising": True,
        "max_iter": 5000,
        "interval": 100,
        "MAX_AD_DIM": 80000,
        # "check_point": f"./h50/focus-init/checkpoint/H50-2.00-oao-mps-rnn-dcut-30-222-focus-20w-checkpoint.pth",
        "prefix": prefix,
        "use_clip_grad": True,
        "max_grad_norm": 1,
        "start_clip_grad": -1,
        "clip_grad_scheduler": clip_grad_scheduler,
    }

* ``nqs``: Ansatz(e.g. **Transformer**, **MPS-RNN**, **Graph-MPS-RNN**).

* ``opt``: Optimizer(e.g., **Adam**, **Adamw**, **SGD**).

* ``lr_scheduler``: LRScheduler, Default: ``None``.

* ``read_model_only``: Read model from the checkpoint file.

* ``dtype``: data-dtype: (e.g., ``Dtype(dtype=torch.complex128, device="cuda")``)

* ``sampler_param``: see :ref:`sample-params`

* ``only_sample``: No calculating gradient. This is used to calculate energy.

* ``max_iter``: the number of the iteration.

* ``interval``: the time of the saving the checkpoint file.

* ``MAX_AD_DIM``: the nbatch of the **backward**.

* ``check_point``: Read model/optimizer/lr_scheduler from the checkpoint file, Default: ``None``.

* ``prefix``: the prefix of the checkpoint file, e.g., ``vmc-checkpoint.pth``.

* ``use_clip_grad``: clip gradient, Default: ``False``.

* ``max_grad_norm``: the max of the l2-norm when clipping gradient.

* ``start_clip_grad``: clip gradient from the k-th iteration.

* ``clip_grad_scheduler``: the scheduler of clipping gradient, this is ``Callable[[int], float]``.

* ``sr``: use minSR.

* ``lm``: use Linear method.

* ``LM_delta``: delta in Linear method.

Optimizer
=========


Linear method
--------------

Linear method ref.

- J. Chem. Phys. 152, 024111 (2020); doi: 10.1063/1.5125803
- PHYSICAL REVIEW RESEARCH 7, 043351 (2025)

Linear method的梯度计算在 ``pynqs/optim/grad/lm.py`` 的函数 ``LM_grad`` 中，
欲使用之，可直接在class ``VMCOptimizer`` 中打开 ``lm: True`` 即可。
除此之外，还有超参 :math:`\delta` 需要调整(即参数 ``LM_delta``)，默认为 :math:`0.1`, 并按照 ``delta = max(delta * 0.9**(epoch), 1e-6)`` 进行衰减。
理论上在优化最后，这项应该衰减至 :math:`0`.

该段代码包含计算梯度和更新两部分，计算梯度按照 J. Chem. Phys. 152, 024111 (2020) 中的方式实现，具体公式推导见文档。
简而言之，最后是构造一下广义本征值问题(GEVP)并求解：

.. math:: 
        Lc = \tilde{E} Rc,\quad 
    L = \begin{bmatrix}
        E & G^\top_{\rm r} \\
        G_{\rm c} & H
    \end{bmatrix} + \delta I
    , 
    R = \begin{bmatrix}
        1&\\
        &S
    \end{bmatrix}+\delta'I

with

.. math:: 
    [G_{\rm r}]_i &= \langle{h_i(n)}\rangle - E\langle{g_i(n)}\rangle\\
    [G_{\rm c}]_i &= \sum_n p(n) \bar{\epsilon}(n) O_i(n)\\
    S_{ij} &= \sum_n p(n) \bar{o}_i(n)O_j(n)\\
    H_{ij} &= \sum_n p(n)\bar{o}_i(n) h_j(n) - [G_{\rm c}]_i\langle{O_j(n)}\rangle

where

.. math::
    \bar{\epsilon}(n) &= E_{\rm loc}(n) - \langle{E_{\rm loc}(n)}\rangle\\
    \bar{o}_i(n) &= O_i(n) - \langle{O_i(n)}\rangle

and

.. math::
    h_i(n) = \partial_i E_{\rm loc}(n) +  O_i(n)E_{\rm loc}(n),\quad \partial_i E_{\rm loc}(n) = \frac{\partial}{\partial\theta_i}\sum_{m\in SD}H_{nm}\frac{\varPsi(m)}{\varPsi(n)}

.. math::
    g_i(n) &= \frac{\langle n|\varPsi_i\rangle}{\langle n|\varPsi\rangle} = \frac{1}{\varPsi(n)} \frac{\partial\varPsi(n)}{\partial\theta_i} = O_i(n) \label{eq:gi}\\
    h_i(n) &= \frac{\langle n|\hat{H}|\varPsi_i\rangle}{\langle n|\varPsi\rangle} = \partial_i E_{\rm loc}(n) +  O_i(n)E_{\rm loc}(n)

这里 :math:`|\varPsi_i\rangle = \partial_{\theta_i}|\varPsi\rangle`,
在更新的时候，实现了以上文章中类似线搜索的方式，见 ``try_step_update`` 函数。