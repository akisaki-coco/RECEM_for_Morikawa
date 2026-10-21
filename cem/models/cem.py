from asyncio import runners
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchvision.models import resnet50

from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils

import torch.nn.functional as F
from torch.autograd import Function

from cem.models.coral import CoralLoss, CoralLayer

# ==============================================================================
# RECEM用 補助モジュール (GRL & HSIC)
# ==============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return -ctx.lambda_ * grads, None

class GRL(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def rbf_kernel(X, sigma=None):
    pairwise_dist = torch.cdist(X, X, p=2) ** 2
    if sigma is None:
        median_dist = torch.median(pairwise_dist.detach())
        sigma = torch.sqrt(0.5 * median_dist + 1e-6)
    gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-8)
    return torch.exp(-gamma * pairwise_dist)

def compute_hsic(Z, C_tilde):
    B = Z.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=Z.device)
    Z_norm = F.normalize(Z, p=2, dim=-1)
    C_norm = F.normalize(C_tilde, p=2, dim=-1)
    K_z = rbf_kernel(Z_norm)
    K_c = rbf_kernel(C_norm)
    H = torch.eye(B, device=Z.device) - (1.0 / B) * torch.ones((B, B), device=Z.device)
    K_z_c = H @ K_z @ H
    K_c_c = H @ K_c @ H
    return torch.trace(K_z_c @ K_c_c) / ((B - 1) ** 2)


################################################################################
## Concept Embedding Models
################################################################################


class ConceptEmbeddingModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,

        # --- ここからRECEM用の引数を追加 ---
        beta_max=0.5,
        beta_hsic=0.1,
        lambda_dis=0.1,
        lambda_rec=0.1,
        grl_lambda=1.0,
        ema_momentum=0.9,

        # CORAL用の追加パラメータ
        use_coral=False,
        num_classes_per_concept=5,
        
    ):
        """
        Constructs a Concept Embedding Model (CEM) as defined by
        Espinosa Zarlenga et al. 2022.

        :param int n_concepts: The number of concepts given at training time.
        :param int n_tasks: The number of output classes of the CEM.
        :param int emb_size: The size of each concept embedding. Defaults to 16.
        :param float training_intervention_prob: RandInt probability. Defaults
            to 0.25.
        :param str embedding_activation: A valid nonlinearity name to use for the
            generated embeddings. It must be one of [None, "sigmoid", "relu",
            "leakyrelu"] and defaults to "leakyrelu".
        :param Bool shared_prob_gen: Whether or not weights are shared across
            all probability generators. Defaults to True.
        :param float concept_loss_weight: Weight to be used for the final loss'
            component corresponding to the concept classification loss. Default
            is 0.01.
        :param float task_loss_weight: Weight to be used for the final loss'
            component corresponding to the output task classification loss.
            Default is 1.

        :param Pytorch.Module c2y_model:  A valid pytorch Module used to map the
            CEM's bottleneck (with size n_concepts * emb_size) to `n_tasks`
            output activations (i.e., the output of the CEM).
            If not given, then a simple leaky-ReLU MLP, whose hidden
            layers have sizes `c2y_layers`, will be used.
        :param List[int] c2y_layers: List of integers defining the size of the
            hidden layers to be used in the MLP to predict classes from the
            bottleneck if c2y_model was NOT provided. If not given, then we will
            use a simple linear layer to map the bottleneck to the output classes.
        :param Fun[(int), Pytorch.Module] c_extractor_arch: A generator function
            for the latent code generator model that takes as an input the size
            of the latent code before the concept embedding generators act (
            using an argument called `output_dim`) and returns a valid Pytorch
            Module that maps this CEM's inputs to the latent space of the
            requested size.

        :param str optimizer:  The name of the optimizer to use. Must be one of
            `adam` or `sgd`. Default is `adam`.
        :param float momentum: Momentum used for optimization. Default is 0.9.
        :param float learning_rate:  Learning rate used for optimization.
            Default is 0.01.
        :param float weight_decay: The weight decay factor used during
            optimization. Default is 4e-05.
        :param List[float] weight_loss: Either None or a list with n_concepts
            elements indicating the weights assigned to each predicted concept
            during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.
        :param List[float] task_class_weights: Either None or a list with
            n_tasks elements indicating the weights assigned to each output
            class during the loss computation. Could be used to improve
            performance/fairness in imbalanced datasets.

        :param List[float] active_intervention_values: A list of n_concepts
            values to use when positively intervening in a given concept (i.e.,
            setting concept c_i to 1 would imply setting its corresponding
            predicted concept to active_intervention_values[i]). If not given,
            then we will assume that we use `1` for all concepts. This
            parameter is important when intervening in CEMs that do not have
            sigmoidal concepts, as the intervention thresholds must then be
            inferred from their empirical training distribution.
        :param List[float] inactive_intervention_values: A list of n_concepts
            values to use when negatively intervening in a given concept (i.e.,
            setting concept c_i to 0 would imply setting its corresponding
            predicted concept to inactive_intervention_values[i]). If not given,
            then we will assume that we use `0` for all concepts.
        :param Callable[(np.ndarray, np.ndarray, np.ndarray), np.ndarray] intervention_policy:
            An optional intervention policy to be used when intervening on a
            test batch sample x (first argument), with corresponding true
            concepts c (second argument), and true labels y (third argument).
            The policy must produce as an output a list of concept indices to
            intervene (in batch form) or a batch of binary masks indicating
            which concepts we will intervene on.

        :param List[int] top_k_accuracy: List of top k values to report accuracy
            for during training/testing when the number of tasks is high.
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        self.output_latent = output_latent
        context_gen_out_size = context_gen_out_size or (2 * emb_size)
        self.pre_concept_model = c_extractor_arch(output_dim=None)
        self._intervention_idxs = None
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)
        self.task_loss_weight = task_loss_weight
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generators = torch.nn.ModuleList()
        self.shared_prob_gen = shared_prob_gen
        self.top_k_accuracy = top_k_accuracy

        # CORAL 用
        self.use_coral = use_coral
        self.num_classes_per_concept = num_classes_per_concept

        # if self.use_coral == True:
        #     # 元概念の数を計算
        #     self.num_orig_concepts = n_concepts // (self.num_classes_per_concept - 1)

        #     # 各元概念に対して，Coralレイヤーを作成
        #     for _ in range(self.num_orig_concepts):
        #         self.concept_prob_generators.append(
        #             CoralLayer(
        #                 in_features=2 * emb_size,
        #                 num_classes=self.num_classes_per_concept
        #             )
        #         )
        
        # 活性化関数の設定 (use_coral の前に定義)
        if embedding_activation is None:
            act_to_use = []
        elif embedding_activation == "sigmoid":
            act_to_use = [torch.nn.Sigmoid()]
        elif embedding_activation == "leakyrelu":
            act_to_use = [torch.nn.LeakyReLU()]
        elif embedding_activation == "relu":
            act_to_use = [torch.nn.ReLU()]
        else:
            raise ValueError(
                f'Unsupported embedding activation "{embedding_activation}"'
            )

        if self.use_coral:
            # 元概念の数を計算 (例: 40 // (5 - 1) = 10)
            self.num_orig_concepts = n_concepts // (self.num_classes_per_concept - 1)

            for _ in range(self.num_orig_concepts):
                # 1. 各元概念専用の特徴量 (16次元) を作るジェネレータ
                self.concept_context_generators.append(
                    torch.nn.Sequential(*([
                        torch.nn.Linear(
                            list(self.pre_concept_model.modules())[-1].out_features,
                            context_gen_out_size,
                        ),
                    ] + act_to_use))
                )
                # 2. 各元概念専用の CoralLayer (重み1つ + バイアス4つ)
                self.concept_prob_generators.append(
                    CoralLayer(
                        in_features=2 * emb_size,
                        num_classes=self.num_classes_per_concept
                    )
                )

        else:
            for i in range(n_concepts):
                self.concept_context_generators.append(
                    torch.nn.Sequential(*([
                        torch.nn.Linear(
                            list(
                                self.pre_concept_model.modules()
                            )[-1].out_features,
                            # Two as each concept will have a positive and a
                            # negative embedding portion which are later mixed
                            context_gen_out_size,
                        ),
                    ] + act_to_use))
                )
                if self.shared_prob_gen and (
                    len(self.concept_prob_generators) == 0
                ):
                    # Then we will use one and only one probability generator which
                    # will be shared among all concepts. This will force concept
                    # embedding vectors to be pushed into the same latent space
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
                elif not self.shared_prob_gen:
                    self.concept_prob_generators.append(torch.nn.Linear(
                        2 * emb_size,
                        1,
                    ))
        if getattr(self, '_construct_c2y_model', True):
            if c2y_model is None:
                # Else we construct it here directly
                units = [
                    n_concepts * emb_size
                ] + (c2y_layers or []) + [n_tasks]
                layers = []
                for i in range(1, len(units)):
                    layers.append(torch.nn.Linear(units[i-1], units[i]))
                    if i != len(units) - 1:
                        layers.append(torch.nn.LeakyReLU())
                self.c2y_model = torch.nn.Sequential(*layers)
            else:
                self.c2y_model = c2y_model
        self.sig = torch.nn.Sigmoid()

        # CORAL用損失関数
        if self.use_coral == True:
            self.loss_concept = CoralLoss(weight=weight_loss)
        else:
            self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.n_tasks = n_tasks
        self.emb_size = emb_size
        self.use_concept_groups = use_concept_groups

        # ConceptEmbeddingModel.__init__ の末尾に追加するコード例
        hidden_dim = list(self.pre_concept_model.modules())[-1].out_features  # バックボーンψの出力次元 (例: ResNetなら512や2048)
        bottleneck_dim = n_concepts * emb_size  # K * d

        # 1. 無関連特徴エンコーダ E_dis: R^{n_hidden} -> R^{K*d}
        self.disentangle_encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LeakyReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )

        # 2. 潜在表現再構成デコーダ D: R^{2*K*d} -> R^{n_hidden}
        # (論文の D(\tilde{C}_i, z_i) に対応。連結 [C_tilde, z_i] または加算を入力とする)
        self.reconstruct_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. 勾配反転層
        self.grl = GRL(lambda_=grl_lambda)

        # 4. 各概念kのアクティブ意味的重心 \bar{c}_k^+ を保持するバッファ [K, d]
        self.register_buffer("semantic_mean_pos", torch.zeros(n_concepts, emb_size))
        self.register_buffer("semantic_mean_initialized", torch.zeros(n_concepts, dtype=torch.bool))
        self.ema_momentum = ema_momentum  # 推論用の重心EMA更新率

        # 損失の重み・スケジュール用ハイパーパラメータ
        self.beta_max = beta_max        # Concept Mixupの最大混合率 (例: 0.5)
        self.beta_hsic = beta_hsic      # HSIC損失の重み (例: 0.1)
        self.lambda_dis = lambda_dis    # 分離損失全体の重み
        self.lambda_rec = lambda_rec    # 再構成損失の重み
        self._recem_cache = {}

    # def _after_interventions(
    #     self,
    #     prob,
    #     pos_embeddings,
    #     neg_embeddings,
    #     intervention_idxs=None,
    #     c_true=None,
    #     train=False,
    #     competencies=None,
    #     **kwargs
    # ):
    #     if train and (self.training_intervention_prob != 0) and (
    #         (c_true is not None) and
    #         (intervention_idxs is None)
    #     ):
    #         # Then we will probabilistically intervene in some concepts
    #         mask = torch.bernoulli(
    #             self.ones * self.training_intervention_prob,
    #         )
    #         intervention_idxs = torch.tile(
    #             mask,
    #             (c_true.shape[0], 1),
    #         )
    #     if (c_true is None) or (intervention_idxs is None):
    #         # Then time to mix!
    #         bottleneck = (
    #             pos_embeddings * torch.unsqueeze(prob, dim=-1) +
    #             neg_embeddings * (1 - torch.unsqueeze(prob, dim=-1))
    #         )
    #         return prob, intervention_idxs, bottleneck
    #     intervention_idxs = intervention_idxs.type(torch.FloatTensor)
    #     intervention_idxs = intervention_idxs.to(prob.device)
    #     output = prob * (1 - intervention_idxs) + intervention_idxs * c_true
    #     # Then time to mix!
    #     bottleneck = self._construct_c2y_input(
    #         pos_embeddings=pos_embeddings,
    #         neg_embeddings=neg_embeddings,
    #         probs=output,
    #         **kwargs,
    #     )
    #     return output, intervention_idxs, bottleneck

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
        **kwargs
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(
                self.ones * self.training_intervention_prob,
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )
        if (c_true is None) or (intervention_idxs is None):
            output = prob
        else:
            intervention_idxs = intervention_idxs.type(torch.FloatTensor)
            intervention_idxs = intervention_idxs.to(prob.device)
            output = prob * (1 - intervention_idxs) + intervention_idxs * c_true

        # 学習時・推論時・介入時すべて共通して _construct_c2y_input (Concept Mixup) を通す
        bottleneck = self._construct_c2y_input(
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            probs=output,
            c_true=c_true,
            train=train,
            **kwargs,
        )
        return output, intervention_idxs, bottleneck

    def _predict_labels(self, bottleneck, **task_loss_kwargs):
        return self.c2y_model(torch.flatten(bottleneck, start_dim=1, end_dim=-1))

    # def _construct_c2y_input(
    #     self,
    #     pos_embeddings,
    #     neg_embeddings,
    #     probs,
    #     **task_loss_kwargs,
    # ):
    #     bottleneck = (
    #         pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
    #             neg_embeddings * (
    #                 1 - torch.unsqueeze(probs, dim=-1)
    #             )
    #         )
    #     )
    #     bottleneck = bottleneck.view(
    #         (-1, self.n_concepts * self.emb_size)
    #     )
    #     return bottleneck

    def _construct_c2y_input(
        self,
        pos_embeddings,
        neg_embeddings,
        probs,
        c_true=None,
        train=False,
        pre_c=None,
        **task_loss_kwargs,
    ):
        B, K, d = pos_embeddings.shape
        probs_exp = torch.unsqueeze(probs, dim=-1)  # [B, K, 1]

        # 1. 通常のCEMの概念埋め込み: c_{i,k} = p_{i,k} * c^+ + (1 - p_{i,k}) * c^-
        c_orig = pos_embeddings * probs_exp + neg_embeddings * (1.0 - probs_exp)

        # 2. Concept Mixup (Semantic Alignment)
        # 学習時は正解ラベル c_true、推論・介入時は予測確率 probs (> 0.5) からアクティブマスク \mu を作成
        if train and (c_true is not None):
            mu = (c_true == 1).float().unsqueeze(-1)  # [B, K, 1]
        else:
            mu = (probs_exp > 0.5).float()            # [B, K, 1]

        active_count = mu.sum(dim=0)  # [K, 1]
        batch_mean_pos = (mu * pos_embeddings).sum(dim=0) / (active_count + 1e-6)  # [K, d]

        if train:
            valid_k = (active_count.squeeze(-1) > 0)
            with torch.no_grad():
                for k in range(K):
                    if valid_k[k]:
                        if not self.semantic_mean_initialized[k]:
                            self.semantic_mean_pos[k] = batch_mean_pos[k]
                            self.semantic_mean_initialized[k] = True
                        else:
                            self.semantic_mean_pos[k] = (
                                self.ema_momentum * self.semantic_mean_pos[k]
                                + (1.0 - self.ema_momentum) * batch_mean_pos[k]
                            )
            c_bar_pos = torch.where(
                valid_k.unsqueeze(-1),
                batch_mean_pos,
                self.semantic_mean_pos.detach(),
            )
            # エポック進行に合わせて混合率 \beta を 0 -> beta_max へスケジュール
            # self.trainer は Trainer 未接続だと RuntimeError を投げるため _trainer を参照
            trainer = getattr(self, "_trainer", None)
            max_epochs = getattr(trainer, "max_epochs", 100) if trainer is not None else 100
            current_epoch = self.current_epoch if trainer is not None else max_epochs
            progress = min(1.0, float(current_epoch) / max(1.0, float(max_epochs)))
            current_beta = self.beta_max * progress
        else:
            c_bar_pos = self.semantic_mean_pos
            current_beta = self.beta_max

        # アクティブ概念のみ意味的重心 \bar{c}_k^+ と混合
        c_bar_expanded = c_bar_pos.unsqueeze(0).expand(B, K, d)
        c_pos_aligned = (1.0 - current_beta) * pos_embeddings + current_beta * c_bar_expanded
        c_active_mixed = probs_exp * c_pos_aligned + (1.0 - probs_exp) * neg_embeddings

        bottleneck = mu * c_active_mixed + (1.0 - mu) * c_orig
        bottleneck = bottleneck.view((-1, self.n_concepts * self.emb_size))

        # 3. 学習時のみ Concept-Level Disentanglement 用の変数を計算してキャッシュ
        if train and (c_true is not None) and (pre_c is not None):
            c_gt_exp = (c_true == 1).float().unsqueeze(-1)
            c_tilde = c_gt_exp * pos_embeddings + (1.0 - c_gt_exp) * neg_embeddings
            c_tilde_flat = c_tilde.view(B, K * d)

            z_i = self.disentangle_encoder(pre_c)
            z_grl = self.grl(z_i)
            y_logits_irr = self.c2y_model(z_grl)
            h_recon = self.reconstruct_decoder(torch.cat([c_tilde_flat, z_i], dim=-1))

            self._recem_cache = {
                "z_i": z_i,
                "c_tilde_flat": c_tilde_flat,
                "y_logits_irr": y_logits_irr,
                "h_recon": h_recon,
                "h_orig": pre_c.detach(),
            }
        else:
            self._recem_cache = {}

        return bottleneck

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        pre_c = None

        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # # CORAL用ロジット・確率生成
            # if self.use_coral == True:
            #     # 10個の元概念コンテキストから，CoralLayerで4次元ずつロジット出力
            #     for i in range(self.num_orig_concepts):

            if self.use_coral:
                for i in range(self.num_orig_concepts):
                    # 概念 i のコンテキスト (16次元) を抽出
                    context = self.concept_context_generators[i](pre_c)
                    # CoralLayer で 4 つの閾値ロジットを出力
                    logits = self.concept_prob_generators[i](context)  # [B, 4]
                    prob = self.sig(logits)                           # [B, 4]
                    c_sem.append(prob)
                    # 4段階の閾値それぞれに共通のコンテキストを割り当て
                    for _ in range(self.num_classes_per_concept - 1):
                        contexts.append(torch.unsqueeze(context, dim=1))
                c_sem = torch.cat(c_sem, dim=-1)      # [B, 40] に結合
                contexts = torch.cat(contexts, dim=1) # [B, 40, 16] に結合

            else:
                # First predict all the concept probabilities
                for i, context_gen in enumerate(self.concept_context_generators):
                    if self.shared_prob_gen:
                        prob_gen = self.concept_prob_generators[0]
                    else:
                        prob_gen = self.concept_prob_generators[i]
                    context = context_gen(pre_c)
                    prob = prob_gen(context)
                    contexts.append(torch.unsqueeze(context, dim=1))
                    c_sem.append(self.sig(prob))
                c_sem = torch.cat(c_sem, dim=-1)
                contexts = torch.cat(contexts, dim=1)

            latent = contexts, c_sem
        else:
            contexts, c_sem = latent

        pos_embeddings = contexts[:, :, :self.emb_size]
        neg_embeddings = contexts[:, :, self.emb_size:]
        # return c_sem, pos_embeddings, neg_embeddings, {}

        # pre_cを他でも使うので返す
        return c_sem, pos_embeddings, neg_embeddings, {"pre_c": pre_c}

    def _new_tail_results(
        self,
        x=None,
        c=None,
        y=None,
        c_sem=None,
        bottleneck=None,
        y_pred=None,
    ):
        return []

    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )

        c_sem, pos_embs, neg_embs, out_kwargs = self._generate_concept_embeddings(
            x=x,
            latent=latent,
            training=train,
        )

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            prior_distribution = self._prior_int_distribution(
                prob=c_sem,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )

        else:
            c_int = c
        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
                device=x.device,
            )

        # Then, time to do the mixing between the positive and the
        # negative embeddings
        probs, intervention_idxs, bottleneck = self._after_interventions(
            c_sem,
            pos_embeddings=pos_embs,
            neg_embeddings=neg_embs,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
            competencies=competencies,
            **out_kwargs
        )
        self._intervention_idxs = intervention_idxs

        y_pred = self._predict_labels(bottleneck=bottleneck)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            if "latent" in out_kwargs:
                latent = (latent or tuple([])) + out_kwargs['latent']
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)

        tail_results += self._new_tail_results(
            x=x,
            c=c,
            y=y,
            c_sem=c_sem,
            bottleneck=bottleneck,
            y_pred=y_pred,
        )
        return tuple([c_sem, bottleneck, y_pred] + tail_results)
    
    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        # 1. 親クラス (ConceptBottleneckModel) の _run_step を実行し、基本のCEM損失を取得
        loss, result = super()._run_step(
            batch=batch,
            batch_idx=batch_idx,
            train=train,
            intervention_idxs=intervention_idxs,
        )

        # 2. 学習時のみ RECEM の分離損失 (L_dis) と再構成損失 (L_rec) を加算
        if train and getattr(self, "_recem_cache", None):
            _, y, _ = self._unpack_batch(batch)
            cache = self._recem_cache

            # (a) 勾配反転層を通した無関連特徴 z_i のタスク分類損失
            if self.n_tasks > 1:
                loss_adv_task = self.loss_task(cache["y_logits_irr"], y.long())
            else:
                loss_adv_task = self.loss_task(cache["y_logits_irr"].squeeze(-1), y.float())

            # (b) z_i と真の概念埋め込み \tilde{C}_i の HSIC 独立性損失
            loss_hsic = compute_hsic(cache["z_i"], cache["c_tilde_flat"])
            loss_dis = loss_adv_task + self.beta_hsic * loss_hsic

            # (c) バックボーン特徴 h_i の再構成損失
            loss_rec = F.mse_loss(cache["h_recon"], cache["h_orig"])

            # 総合損失への加算
            recem_extra_loss = (self.lambda_dis * loss_dis) + (self.lambda_rec * loss_rec)
            loss = loss + recem_extra_loss

            # ログ確認用に result 辞書にも記録
            result["loss_dis"] = loss_dis.detach().item()
            result["loss_hsic"] = loss_hsic.detach().item()
            result["loss_rec"] = loss_rec.detach().item()
            result["loss"] = loss.detach().item()

            # メモリリーク防止のためキャッシュをクリア
            self._recem_cache = {}

        return loss, result



################################################################################
## Fixed Embedding Version
################################################################################


class FixedEmbConceptEmbeddingModel(ConceptEmbeddingModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        shared_prob_gen=True,
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        context_gen_out_size=None,

        top_k_accuracy=None,

        # New parameters
        fixed_embeddings=True,
        initial_concept_embeddings=None,
        fixed_embeddings_always=True,

        # # CORALを使うなら使うフラグ
        # use_coral=False,
        # # 各概念のグレード数 (今回は5段階評価)
        # num_classes_per_concept=5,
        
    ):
        """
        Same as a CEM but it has a set of learnable global embeddings
        to use for each concept. Useful if you want the interventions
        to be done using a global set of embeddings.
        """

        super(FixedEmbConceptEmbeddingModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=shared_prob_gen,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=None, # KEY!!!!
            inactive_intervention_values=None, # KEY!!!!
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            use_concept_groups=use_concept_groups,
            context_gen_out_size=context_gen_out_size,
            top_k_accuracy=top_k_accuracy,
        )

        # Let's generate the global embeddings we will use
        if (
            (initial_concept_embeddings is None) and
            (active_intervention_values is not None) and
            (inactive_intervention_values is not None)
        ):
            active_intervention_values = torch.tensor(
                active_intervention_values
            )
            inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )

            initial_concept_embeddings = torch.concat(
                [
                    active_intervention_values.unsqueeze(1),
                    inactive_intervention_values.unsqueeze(1),
                ],
                dim=1,
            )
        self.fixed_embeddings_always = fixed_embeddings_always
        self._set_embeddings = True
        if (initial_concept_embeddings is False) or (
            initial_concept_embeddings is None
        ):
            initial_concept_embeddings = torch.normal(
                torch.zeros(self.n_concepts, 2, emb_size),
                torch.ones(self.n_concepts, 2, emb_size),
            )
        else:
            if isinstance(initial_concept_embeddings, np.ndarray):
                initial_concept_embeddings = torch.FloatTensor(
                    initial_concept_embeddings
                )
        self.concept_embeddings = torch.nn.Parameter(
            initial_concept_embeddings,
            requires_grad=(not fixed_embeddings),
        )

    def _generate_concept_embeddings(
        self,
        x,
        latent=None,
        training=False,
    ):
        if not self._set_embeddings:
            # Then run the standard CEM pathway
            return ConceptEmbeddingModel._generate_concept_embeddings(
                self=self,
                x=x,
                latent=latent,
                training=training,
            )
        if latent is None:
            pre_c = self.pre_concept_model(x)
            contexts = []
            c_sem = []

            # First predict all the concept probabilities
            for i, context_gen in enumerate(self.concept_context_generators):
                if self.shared_prob_gen:
                    prob_gen = self.concept_prob_generators[0]
                else:
                    prob_gen = self.concept_prob_generators[i]
                context = context_gen(pre_c)
                prob = prob_gen(context)
                contexts.append(torch.unsqueeze(context, dim=1))
                c_sem.append(self.sig(prob))
            c_sem = torch.cat(c_sem, dim=-1)
            contexts = torch.cat(contexts, dim=1)
            latent = contexts, c_sem
        else:
            contexts, c_sem = latent
        if self.fixed_embeddings_always:
            pos_embeddings = self.concept_embeddings[:, 0, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
            neg_embeddings = self.concept_embeddings[:, 1, :].unsqueeze(0).expand(
                x.shape[0],
                -1,
                -1,
            )
        else:
            # Else we only use fixed embeddings for interventions
            pos_embeddings = contexts[:, :, :self.emb_size]
            neg_embeddings = contexts[:, :, self.emb_size:]
        return c_sem, pos_embeddings, neg_embeddings, {}
