#!/bin/bash
# LoRA train script by @Akegarasu modify by @bdsqlsz

# Train Model | 训练模式
model="sdxl_db" #lora、db、sdxl_lora、sdxl_db、contralnet(unfinished)

# Train data path | 设置训练用模型、图片
pretrained_model="/root/autodl-tmp/models/ckpts/sdxlFixedvaeFp16Remove_baseFxiedVaeV2Fp16.safetensors" # base model path | 底模路径
is_v2_model=0                             # SD2.0 model | SD2.0模型 2.0模型下 clip_skip 默认无效
v_parameterization=0 # parameterization | 参数化 v2 非512基础分辨率版本必须使用。
vae=""
train_data_dir="/root/autodl-tmp/tile_paint"            # train dataset path | 训练数据集路径
reg_data_dir=""      # directory for regularization images | 正则化数据集路径，默认不使用正则化图像。
training_comment="this_LoRA_model_credit_from_bdsqlsz"	# training_comment | 训练介绍，可以写作者名或者使用触发关键词

# Train related params | 训练相关参数
resolution="1024,1024"  # image resolution w,h. 图片分辨率，宽,高。支持非正方形，但必须是 64 倍数。
batch_size=12        # batch size
vae_batch_size=4 #vae初始化转换图片批处理大小，2-4。大了可以让一开始处理图片更快
max_train_epoches=36  # max train epoches | 最大训练 epoch
save_every_n_epochs=6 # sa2ve every n epochs | 每 N 个 epoch 保存一次

gradient_checkpointing=1 #梯度检查，开启后可节约显存，但是速度变慢
gradient_accumulation_steps=0 # 梯度累加数量，变相放大batchsize的倍数

network_dim=32   # network dim | 常用 4~128，不是越大越好
network_alpha=1 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。

#dropout | 抛出(目前和lycoris不兼容，请使用lycoris自带dropout)
network_dropout="0.1" # dropout 是机器学习中防止神经网络过拟合的技术，建议0.1~0.3 
scale_weight_norms="1.0" #配合 dropout 使用，最大范数约束，推荐1.0
rank_dropout="0" #lora模型独创，rank级别的dropout，推荐0.1~0.3，未测试过多
module_dropout="0" #lora模型独创，module级别的dropout(就是分层模块的)，推荐0.1~0.3，未测试过多
caption_dropout_rate="0.1"

train_unet_only=1         # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
train_text_encoder_only=0 # train Text Encoder only | 仅训练 文本编码器

seed="1026" # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

noise_offset="0.0375" # noise offset | 在训练中添加噪声偏移来改良生成非常暗或者非常亮的图像，如果启用，推荐参数为0.1
adaptive_noise_scale="0.0375" #adaptive noise scale | 自适应噪声偏移范围
multires_noise_iterations="0" #多分辨率噪声扩散次数，推荐6-10,0禁用,和noise_offset冲突，只能开一个
multires_noise_discount="0" #多分辨率噪声缩放倍数，推荐0.1-0.3,上面关掉的话禁用。

shuffle_caption=1 # 随机打乱tokens顺序，默认启用。修改为 0 禁用。
keep_tokens=3  # keep heading N tokens when shuffling caption tokens | 在随机打乱 tokens 时，保留前 N 个不变。

prior_loss_weight=1 #正则化权重，0-1

# Learning rate | 学习率
lr="5e-6"
unet_lr="1e-5"
text_encoder_lr="2e-6"
lr_scheduler="constant_with_warmup" # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
lr_warmup_steps=10                   # warmup steps | 仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
lr_restart_cycles=1                 # cosine_with_restarts restart cycles | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时起效。

min_snr_gamma=5 #最小信噪比伽马值，减少低step时loss值，让学习效果更好。推荐3-5，5对原模型几乎没有太多影响，3会改变最终结果。修改为0禁用。
weighted_captions=0 #权重打标，默认识别标签权重，语法同webui基础用法。例如(abc), [abc], (abc:1.23),但是不能再括号内加逗号，否则无法识别。

# Merge lora and train | 差异提取法
base_weights=""
base_weights_multiplier="1.0"

# Block weights | 分层训练
enable_block_weights=0 #开启分层训练
down_lr_weight="1,0.2,1,1,0.2,1,1,0.2,1,1,1,1" #12层，需要填写12个数字，0-1.也可以使用函数写法，支持sine, cosine, linear, reverse_linear, zeros，参考写法down_lr_weight=cosine+.25 
mid_lr_weight="1"  #1层，需要填写1个数字，其他同上。
up_lr_weight="1,1,1,1,1,1,1,1,1,1,1,1"   #12层，同上上。
block_lr_zero_threshold=0  #如果分层权重不超过这个值，那么直接不训练。默认0。

enable_block_dim=0 #开启分块dim训练
block_dims="64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64" #dim分块，25块
block_alphas="1,1,2,1,2,2,4,1,1,4,4,4,1,4,1,4,2,1,1,4,1,1,1,4,1"  #alpha分块，25块
conv_block_dims="32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32" #convdim分块，25块
conv_block_alphas="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" #convalpha分块，25块

# Output settings | 输出设置
output_name="SCAnime"           # output model name | 模型保存名称
save_model_as="safetensors" # model save ext | 模型保存格式 ckpt, pt, safetensors
mixed_precision="bf16" # bf16效果更好但是30系以下显卡不支持，默认fp16
save_precision="bf16" # bf16效果更好但是30系以下显卡不支持，默认fp16
full_fp16=0 # 半精度全部使用fp16
full_bf16=1 # 半精度全部使用bf16
cache_latents=1 #缓存潜变量
cache_latents_to_disk=1 #开启缓存潜变量保存到磁盘，这样下次训练不用再次缓存转换，速度更快
no_half_vae=0 #禁止半精度，防止黑图。无法和mixed_precision混合精度共用。

# wandb 
wandb_api_key=""
log_tracker_name=$output_name 

# Sample output | 出图
enable_sample=1 #开启出图
sample_every_n_epochs=2 #每n个epoch出一次图
sample_prompts="/root/autodl-tmp/anime.txt"
sample_sampler="euler_a"

# 其他设置
network_weights=""               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。
enable_bucket=1 # arb for diff wh | 分桶
min_bucket_reso=512              # arb min resolution | arb 最小分辨率
max_bucket_reso=1536             # arb max resolution | arb 最大分辨率
persistent_data_loader_workers=1 # persistent dataloader workers | 容易爆内存，保留加载训练集的worker，减少每个 epoch 之间的停顿
clip_skip=2                      # clip skip | 玄学 一般用 2
save_state=1 #save
resume=""

# 优化器设置
#use_8bit_adam=1 # use 8bit adam optimizer | 使用 8bit adam 优化器节省显存，默认启用。部分 10 系老显卡无法使用，修改为 0 禁用。
#use_lion=0      # use lion optimizer | 使用 Lion 优化器
optimizer_type="PagedAdamW8bit" # "adaFactor","AdamW8bit","Lion","DAdaptation",  推荐新优化器Lion。推荐学习率unetlr=lr=6e-5,tenclr=7e-6
# 新增优化器"Lion8bit"(速度更快，内存消耗更少)、"DAdaptAdaGrad"、"DAdaptAdan"(北大最新算法，效果待测)、"DAdaptSGD"
# 新增优化器 Sophia(2倍速1.7倍显存)、Prodigy天才优化器，可自适应Dylora
d0="4e-7"


# lycoris 训练设置
enable_lycoris_train=0 # enable lycoris train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
conv_dim=8           # conv dim | 类似于 network_dim，推荐为 4
conv_alpha=1         # conv alpha | 类似于 network_alpha，可以采用与 conv_dim 一致或者更小的值
algo="loha" # algo参数，制定训练lycoris模型种类，包括lora(locon)、loha、IA3以及lokr、dylora 。5个可选
dropout="0" #lycoris专用dropout

# dylora 训练设置
enable_dylora_train=0 # enable dylora train | 启用 LoCon 训练 启用后 network_dim 和 network_alpha 应当选择较小的值，比如 2~16
unit=4	#block size

# SDXL 
min_timestep="0" #最小时序，默认值0
max_timestep="1000" #最大时序，默认值1000
bucket_reso_steps="32" #default 64,SDXL can use 32
cache_text_encoder_outputs=0 #开启缓存文本编码器，开启后减少显存使用。但是无法和shuffle共用
cache_text_encoder_outputs_to_disk=0 #开启缓存文本编码器到磁盘，开启后减少显存使用。但是无法和shuffle共用

#checkpoint train
no_token_padding=1 #不进行分词器填充

diffuser_xformers=0
train_text_encoder=0



# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

network_module="networks.lora"
extArgs=()
train_script="train_network"

if [ $no_half_vae == 1 ]; then 
  extArgs+=("--no_half_vae"); 
  mixed_precision="no";
  full_bf16=0
  full_fp16=0
fi

if [[ $model == *db ]] ; then
	if [[ $model == "db" ]] ; then 
		train_script="train_db";
		if [[ $no_token_padding -ne 0 ]]; then extArgs+=("--no_token_padding"); fi
	else 
		train_script="train";
		if [[ $diffuser_xformers -ne 0 ]]; then extArgs+=("--diffuser_xformers"); fi
		if [[ $train_text_encoder -ne 0 ]]; then extArgs+=("--train_text_encoder"); fi
	fi
	network_module=""
	network_dim=""
	network_alpha=""
	network_weights=""
	enable_block_weights=0
	enable_block_dim=0
	enable_lycoris_train=0
	enable_dylora_train=0
	unet_lr=""
	text_encoder_lr=""
	train_unet_only=0
	train_text_encoder_only=0
	training_comment=""
	prior_loss_weight=1
	network_dropout="0"
fi

if [[ $model == sdxl* ]] ; then
	train_script="sdxl_$train_script";
	if [ $cache_text_encoder_outputs -ne 0 ]; then 
  		extArgs+=("--cache_text_encoder_outputs"); 
  		enable_bucket=0
  		shuffle_caption=0
  		if [ $cache_text_encoder_outputs_to_disk -ne 0 ]; then 
  			extArgs+=("--cache_text_encoder_outputs_to_disk"); 
		fi
	fi
	
	if [ $full_bf16 -ne 0 ]; then 
  		extArgs+=("--full_bf16"); 
  		mixed_precision="bf16"
  		full_fp16=0
	fi
	
	if [ $bucket_reso_steps != "64" ]; then 
  		extArgs+=("--bucket_reso_steps=$bucket_reso_steps"); 
	fi
	
	if [ $min_timestep != "0" ]; then 
  		extArgs+=("--min_timestep=$min_timestep"); 
	fi

	if [ $max_timestep != "1000" ]; then 
  		extArgs+=("--max_timestep=$max_timestep"); 
	fi
fi

if [ $network_module ]; then extArgs+=("--network_module=$network_module"); fi

if [ $network_dim ]; then
  extArgs+=("--network_dim=$network_dim")
  if [ $network_alpha ]; then
      extArgs+=("--network_alpha=$network_alpha")
  fi
fi

if [ $unet_lr ]; then extArgs+=("--unet_lr=$unet_lr"); fi

if [ $text_encoder_lr ]; then extArgs+=("--text_encoder_lr=$text_encoder_lr"); fi

if [ $prior_loss_weight -ne 1 ]; then extArgs+=("--prior_loss_weight=$prior_loss_weight"); fi

if [ $training_comment ]; then extArgs+=("--training_comment=$training_comment"); fi

if [ $save_state ]; then extArgs+=("--save_state"); fi

if [ $resume ]; then extArgs+=("--resume=$resume"); fi

if [ $reg_data_dir ]; then extArgs+=("--reg_data_dir=$reg_data_dir"); fi

if [ $train_unet_only -ne 0 ]; then extArgs+=("--network_train_unet_only"); 
elif [ $train_text_encoder_only -ne 0 ]; then extArgs+=("--network_train_text_encoder_only"); 
fi

if [ $network_weights ]; then extArgs+=("--network_weights=$network_weights"); fi

if [ $reg_data_dir ]; then extArgs+=("--reg_data_dir=$reg_data_dir"); fi

if [ $shuffle_caption -ne 0 ]; then extArgs+=("--shuffle_caption"); fi

if [ $persistent_data_loader_workers -ne 0 ]; then extArgs+=("--persistent_data_loader_workers"); fi

if [ $weighted_captions -ne 0 ]; then extArgs+=("--weighted_captions"); fi

if [ $caption_dropout_rate != "0" ]; then extArgs+=("--caption_dropout_rate=$caption_dropout_rate"); fi

if [ $vae ]; then extArgs+=("--vae=$vae"); fi

if [ $cache_latents -ne 0 ]; then 
	extArgs+=("--cache_latents"); 
	if [ $cache_latents_to_disk -ne 0 ]; then 
		extArgs+=("--cache_latents_to_disk"); 
	fi
fi

if [ $full_fp16 -ne 0 ]; then 
  extArgs+=("--full_fp16"); 
  mixed_precision="fp16";
fi

if [ $mixed_precision != "no" ]; then 
  extArgs+=("--mixed_precision=$mixed_precision"); 
fi

if [[ $network_dropout != "0" ]]; then
  enable_lycoris=0
  extArgs+=("--network_dropout=$network_dropout"); 
  extArgs+=("--scale_weight_norms=$scale_weight_norms"); 
  if [[ $enable_dylora != "0" && $model != db* ]]; then
    extArgs+=("--network_args rank_dropout=$rank_dropout module_dropout=$module_dropout")
  fi
fi

if [ $enable_lycoris_train == 1 ]; then
  network_module="lycoris.kohya"
  extArgs+=("--network_args conv_dim=$conv_dim conv_alpha=$conv_alpha algo=$algo dropout=$dropout")

elif [ $enable_dylora_train == 1 ]; then
  network_module="networks.dylora"
  extArgs+=("--network_args unit=$unit")
  if [[ $module_dropout != "0" ]]; then
    extArgs+=("module_dropout=$module_dropout")
  fi

elif [ $enable_block_weights == 1 ]; then
  extArgs+=("--network_args down_lr_weight=$down_lr_weight mid_lr_weight=$mid_lr_weight up_lr_weight=$up_lr_weight block_lr_zero_threshold=$block_lr_zero_threshold")
  if [ $enable_block_dim == 1 ]; then
    extArgs+=("block_dims=$block_dims block_alphas=$block_alphas")
    if [ $conv_block_dims ]; then
      extArgs+=("conv_block_dims=$conv_block_dims conv_block_alphas=$conv_block_alphas")
    fi
  fi
fi

if [[ $optimizer_type == "Lion" ]] ; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.95,.98")

elif [[ $optimizer_type == "DAdaptation" ]] || [[ $optimizer_type == "DAdaptAdam" ]] ; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 decouple=True use_bias_correction=True")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"

elif [[ $optimizer_type == "DAdaptAdan" ]] || [[ $optimizer_type == "DAdaptSGD" ]] || [[ $optimizer_type == "DAdaptAdaGrad" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.965,.95,.98")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"
  
elif [[ $optimizer_type == "adafactor" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args scale_parameter=False warmup_init=False relative_step=False")
  
elif [[ $optimizer_type == "Prodigy" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 decouple=True use_bias_correction=True d_coef=1.0 d0=$d0 safeguard_warmup=True")
  lr="1"
  unet_lr="1"
  text_encoder_lr="1"


elif [[ $optimizer_type == "PagedAdamW8bit" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 ")


elif [[ $optimizer_type == "AdamW8bit" ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 is_paged=True")

elif [[ $optimizer_type == *Lion8bit ]]; then
  extArgs+=("--optimizer_type=$optimizer_type" "--optimizer_args weight_decay=0.01 betas=.95,.98 is_paged=True")
fi

if [[ $noise_offset != "0" ]]; then 
  extArgs+=("--noise_offset=$noise_offset"); 
  if [[ $adaptive_noise_scale != "0" ]]; then extArgs+=("--adaptive_noise_scale=$adaptive_noise_scale"); fi  
elif [[ $multires_noise_iterations != "0" ]]; then 
  extArgs+=("--multires_noise_iterations=$multires_noise_iterations"); 
  extArgs+=("--multires_noise_discount=$multires_noise_discount"); 
fi

if [[ $vae_batch_size -ne 0 ]]; then extArgs+=("--vae_batch_size=$vae_batch_size"); fi

if [[ $min_snr_gamma -ne 0 ]]; then extArgs+=("--min_snr_gamma=$min_snr_gamma"); fi

if [[ $gradient_checkpointing -ne 0 ]]; then extArgs+=("--gradient_checkpointing"); fi

if [[ $gradient_accumulation_steps -ne 0 ]]; then extArgs+=("--gradient_accumulation_steps=$gradient_accumulation_steps"); fi

if [[ $is_v2_model == 1 ]]; then
  extArgs+=("--v2");
  extArgs+=("--v_parameterization");
  extArgs+=("--scale_v_pred_loss_like_noise_pred");
else
  extArgs+=("--clip_skip=$clip_skip");
fi

if [ $wandb_api_key ]; then
  extArgs+=("--wandb_api_key=$wandb_api_key");
  extArgs+=("--log_with=wandb");
  extArgs+=("--log_tracker_name=$log_tracker_name");
fi

if [ $enable_sample == 1 ]; then
  extArgs+=("--sample_every_n_epochs=$sample_every_n_epochs");
  extArgs+=("--sample_prompts=$sample_prompts");
  extArgs+=("--sample_sampler=$sample_sampler");
fi

if [[ $enable_bucket -ne 0 ]]; then 
  extArgs+=("--enable_bucket"); 
  if [[ $min_bucket_reso != "0" ]]; 
  	then extArgs+=("--min_bucket_reso=$min_bucket_reso"); 
  fi
  if [[ $max_bucket_reso != "0" ]]; 
  	then extArgs+=("--max_bucket_reso=$max_bucket_reso"); 
  fi
fi  

accelerate launch --num_cpu_threads_per_process=8 "./$train_script.py" \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --resolution=$resolution \
  --max_train_epochs=$max_train_epoches \
  --learning_rate=$lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_num_cycles=$lr_restart_cycles \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --save_precision=$save_precision \
  --seed=$seed \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  --keep_tokens=$keep_tokens \
  --xformers \
  ${extArgs[@]}
