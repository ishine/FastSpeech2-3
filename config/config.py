from pathlib import Path  # 导入Path类用于路径操作
from dataclasses import dataclass, field  # 导入dataclass和field用于创建数据类
from typing import Optional, Tuple, List, Union  # 导入类型提示相关模块

@dataclass
class TrainConfig:
    # 预处理相关配置
    n_threads: int = 16  # 用于并行处理语音数据的线程数
    include_empty_intervals: bool = True  # 是否加载静音区间

    mel_fmin: int = 0  # 梅尔频谱的最低频率
    mel_fmax: int = 8000  # 梅尔频谱的最高频率
    hop_length: int = 192  # STFT的跳帧长度
    stft_length: int = 768  # STFT的窗口长度
    sample_rate: int = 16000  # 采样率
    window_length: int = 768  # 窗口长度
    n_mel_channels: int = 80  # 梅尔频谱通道数

    raw_data_path: Path = "data/ssw_esd"  # 原始数据路径
    val_ids_path: Path = "data/val_ids.txt"  # 验证集ID文件路径
    test_ids_path: Path = "data/test_ids.txt"  # 测试集ID文件路径
    preprocessed_data_path: Path = Path("data/preprocessed")  # 预处理数据路径

    egemap_feature_names: Tuple[str] = (  # Egemap特征名称
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "spectralFlux_sma3_amean",
        "HNRdBACF_sma3nz_amean",
        "mfcc1V_sma3nz_amean",
        "equivalentSoundLevel_dBp"
    )

    # Vocoder配置
    vocoder_checkpoint_path: str = "data/g_01800000"  # Vocoder检查点路径
    istft_resblock_kernel_sizes: Tuple[int] = (3, 7, 11)  # iSTFT残差块卷积核大小
    istft_upsample_rates: Tuple[int] = (6, 8)  # iSTFT上采样率
    istft_upsample_initial_channel: int = 512  # iSTFT上采样初始通道数
    istft_upsample_kernel_sizes: Tuple[int] = (16, 16)  # iSTFT上采样卷积核大小
    istft_resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))  # iSTFT残差块扩张大小
    gen_istft_n_fft: int = 16  # iSTFT的FFT点数
    gen_istft_hop_size: int = 4  # iSTFT的跳帧长度

    # Transformer编码器配置
    padding_index: int = 0  # 填充索引
    max_seq_len: int = 2000  # 最大序列长度
    phones_mapping_path: Path = Path("data/preprocessed/phones.json")  # 音素映射文件路径
    transformer_encoder_hidden: int = 512  # Transformer编码器隐藏层大小
    transformer_encoder_layer: int = 9  # Transformer编码器层数
    transformer_encoder_head: int = 2  # Transformer编码器头数
    transformer_conv_filter_size: int = 512  # Transformer卷积过滤器大小
    transformer_conv_kernel_size: tuple = (9, 1)  # Transformer卷积核大小
    transformer_encoder_dropout: float = 0.2  # Transformer编码器Dropout率

    # Transformer解码器配置
    transformer_decoder_hidden: int = 512  # Transformer解码器隐藏层大小
    transformer_decoder_layer: int = 9  # Transformer解码器层数
    transformer_decoder_head: int = 2  # Transformer解码器头数
    transformer_decoder_dropout: float = 0.2  # Transformer解码器Dropout率

    # 情感条件配置
    emotion_emb_hidden_size: int = 256  # 情感嵌入隐藏层大小
    stack_speaker_with_emotion_embedding: bool = True  # 是否将说话人和情感嵌入连接
    n_egemap_features: int = 2  # Egemap特征数
    conditional_layer_norm: bool = True  # 是否使用条件层归一化
    conditional_cross_attention: bool = True  # 是否使用条件交叉注意力

    # 判别器配置
    compute_adversarial_loss: bool = True  # 是否计算对抗损失
    compute_fm_loss: bool = True  # 是否计算FM损失
    optimizer_lrate_d: float = 1e-4  # 判别器优化器学习率
    optimizer_betas_d: tuple[float, float] = (0.5, 0.9)  # 判别器优化器betas
    kernels_d: tuple[float, ...] = (3, 5, 5, 5, 3)  # 判别器卷积核大小
    strides_d: tuple[float, ...] = (1, 2, 2, 1, 1)  # 判别器步幅

    # FastSpeech2，方差预测器配置
    speaker_emb_hidden_size: int = 256  # 说话人嵌入隐藏层大小
    variance_embedding_n_bins: int = 256  # 方差嵌入bin数
    variance_predictor_kernel_size: int = 3  # 方差预测器卷积核大小
    variance_predictor_filter_size: int = 256  # 方差预测器过滤器大小
    variance_predictor_dropout: float = 0.5  # 方差预测器Dropout率

    # 数据集配置
    multi_speaker: bool = True  # 是否使用多说话人
    multi_emotion: bool = True  # 是否使用多情感
    n_emotions: int = 5  # 情感数
    n_speakers: int = 10  # 说话人数
    train_batch_size: int = 64  # 训练批量大小
    val_batch_size: int = 32  # 验证批量大小
    device: str = "cuda"  # 设备类型

    # 训练配置
    seed: int = 55  # 随机种子
    precision: str = 32  # 精度
    matmul_precision: str = "high"  # 矩阵乘法精度
    lightning_checkpoint_path: str = "data/checkpoint"  # 检查点路径
    train_from_checkpoint: Optional[str] = None  # 从检查点恢复训练
    num_workers: int = 1  # 数据加载器线程数
    test_wav_files_directory: str = "data/wav"  # 测试wav文件目录
    test_mos_files_directory: str = "data/mos"  # 测试MOS文件目录
    total_training_steps: int = 50000  # 总训练步数
    val_each_epoch: int = 20  # 每个epoch验证次数
    val_audio_log_each_step: int = 1  # 每多少步记录一次音频日志

    # 测试/推理配置
    testing_checkpoint: str = "data/emospeech.ckpt"  # 测试检查点路径
    audio_save_path: str = "data/deepvk_test"  # 合成音频保存路径
    nisqa_save_path: str = "data/deepvk_test"  # NISQA输出文件保存路径
    limit_generation: int = None  # 限制生成样本数
    compute_nisqa_on_test: bool = True  # 是否在测试时计算NISQA分数
    phones_path: str = "data/phones.json"  # 音素字典路径

    # 优化器配置
    optimizer_grad_clip_val: float = 1.0  # 优化器梯度裁剪值
    optimizer_warm_up_step: float = 4000  # 优化器预热步数
    optimizer_anneal_steps: tuple[float, ...] = (300000, 400000, 500000)  # 优化器退火步数
    optimizer_anneal_rate: float = 0.3  # 优化器退火率
    fastspeech_optimizer_betas: tuple[float, float] = (0.9, 0.98)  # FastSpeech优化器betas
    fastspeech_optimizer_eps: float = 1e-9  # FastSpeech优化器eps
    fastspeech_optimizer_weight_decay: float = 0.0  # FastSpeech优化器权重衰减

    # Wandb配置
    wandb_log_model: bool = False  # 是否记录模型到Wandb
    wandb_project: str = "EmoSpeech"  # Wandb项目名
    wandb_run_id: str = None  # Wandb运行ID
    resume_wandb_run: bool = False  # 是否恢复Wandb运行
    strategy: str = "ddp_find_unused_parameters_true"  # 分布式训练策略
    wandb_offline: bool = False  # 是否离线记录Wandb日志
    wandb_progress_bar_refresh_rate: int = 1  # Wandb进度条刷新率
    wandb_log_every_n_steps: int = 1  # 每多少步记录一次Wandb日志
    devices: Union[tuple, int] = (0, 1, 2, 3)  # 设备列表或设备数
    limit_val_batches: Optional[int] = 4  # 验证批次限制
    limit_test_batches: Optional[int] = 4  # 测试批次限制
    num_sanity_val_steps: int = 4  # 验证步骤数
    save_top_k_model_weights: int = 5  # 保存最优模型权重数
    metric_monitor_mode: str = "max"  # 指标监控模式

    def __post_init__(self):
        self.hop_in_ms = self.hop_length / self.sample_rate  # 计算hop长度（毫秒）
        if self.stack_speaker_with_emotion_embedding:
            self.emb_size_dis = self.speaker_emb_hidden_size + self.emotion_emb_hidden_size  # 计算判别器嵌入大小
        else:
            self.emb_size_dis = self.emotion_emb_hidden_size  # 判别器嵌入大小等于情感嵌入隐藏层大小
