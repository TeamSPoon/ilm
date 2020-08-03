DATASET=roc_stories
TRAIN_DIR=/apdcephfs/share_916081/shared_info/rickwwang_ckpt/log_senlm
EXAMPLES_DIR=data/sen_masks/${DATASET}
CUDA_VISIBLE_DEVICES=6 python train_ilm.py \
	experiment_${DATASET} \
	${TRAIN_DIR} \
	${EXAMPLES_DIR} \
	--seed 0 \
	--train_examples_tag train \
	--eval_examples_tag valid \
	--train_batch_size 24 \
	--train_batch_accumulation 1 \
	--eval_batch_size 8 \
    --mask_cls ilm.mask.mid_sentence.MaskMSentence \
    --wandb
#    --train_from_scratch