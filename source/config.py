
import datetime
import os
import threading


class Config(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self):
        if not Config._init_flag:
            Config._init_flag = True
            root_path = str(os.getcwd()).replace("\\", "/")
            if 'source' in root_path.split('/'):
                self.base_path = os.path.abspath(os.path.join(os.path.pardir))
            else:
                self.base_path = os.path.abspath(os.path.join(os.getcwd(), ''))
            self._init_train_config()

    def __new__(cls, *args, **kwargs):
        """
        单例类
        :param args:
        :param kwargs:
        :return:
        """
        if not hasattr(Config, '_instance'):
            with Config._instance_lock:
                if not hasattr(Config, '_instance'):
                    Config._instance = object.__new__(cls)
        return Config._instance

    def _init_train_config(self):
        self.label_list = []
        self.use_gpu = True
        self.device = "cpu"
        self.sep = " "

        # 输入数据集、输出目录 (指向 data/step1_base 文件夹)
        # 混合了真实噪声的训练集 (供模型在泥潭里学习)
        self.train_file = os.path.join(self.base_path, 'data', 'final_datasets', 'Dataset2.txt')

        # 绝对干净、互不重叠的验证集 (供模型每轮自测找最高分)
        self.eval_file = os.path.join(self.base_path, 'data', 'final_datasets', 'eval.txt')

        # 绝对干净的期末考卷 (留给论文最终跑指标用的)
        self.test_file = os.path.join(self.base_path, 'data', 'final_datasets', 'test.txt')

        # 建议：让输出的模型也保存在专属的 step1_base 文件夹下，避免跟未来的正是实验搞混
        self.output_path = os.path.join(
            self.base_path, 'output', 'step2_main', datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        )
        # Pretrained model name or path if not the same as model_name
        self.model_name_or_path = os.path.join(self.base_path, 'bert-base-NER')

        self.do_train = True
        self.do_eval = True
        self.do_test = False
        self.clean = True
        self.logging_steps = 50

        # ========================================================
        # 核心多约束损失权重 (根据论文 Eq. 21 和 22)
        # ========================================================
        self.loss_alpha = 0.1  # a: 自监督分类损失的权重 (冷启动阶段设为1.0)
        self.loss_beta = 0.3  # b: 交叉熵分类损失的权重 (论文超参数敏感性分析得出0.3最优)
        self.loss_gamma = 0.7  # r: 标签分布学习(LDL)与一致性损失的权重 (随阶段提升)

        # ========================================================
        # 噪声过滤与标签分布学习(LDL)参数
        # ========================================================
        self.mixup_alpha = 0.4  # SeqMix数据增强中 Beta 分布的 alpha 参数
        self.ood_noise_ratio = 0.01  # 分布外(OOD)噪声过滤比例：假设高损失集合中密度最低的 10% 为 OOD
        self.kde_bandwidth = 1.0  # 核密度估计(KDE)的基础带宽 (h_x, h_y)

        # ========================================================
        # 协同训练机制控制参数
        # ========================================================
        self.drop_rate = 0.5  # 初始的 drop rate
        self.final_drop_rate = 0.1  # 最终的 drop rate
        self.stage1 = 5  # 冷启动阶段的 epoch 数量 (论文设置前5个epoch为第一阶段)

        # ========================================================
        # 模型与训练参数 (根据论文 4.2 实验设置对齐)
        # ========================================================
        self.need_birnn = True
        self.do_lower_case = True
        self.rnn_dim = 128  # 隐藏层大小 (论文指定 128)
        self.hidden_dropout_prob = 0.5  # 论文指定的 Dropout 比例 0.5

        self.max_seq_length = 128  # 样本最大序列长度
        self.train_batch_size = 16  # 论文指定 Batch Size 为 32
        self.eval_batch_size = 32
        self.num_train_epochs = 20  # 论文指定总训练 Epoch 为 20
        self.epochs = 40  # (兼容原代码调度器长度)

        self.gradient_accumulation_steps = 2
        self.learning_rate = 1e-5  # 论文指定学习率 3e-5
        self.weight_decay = 9e-3  # 论文指定 Weight Decay 为 9e-3
        self.adam_epsilon = 1e-8
        self.warmup_steps = 360