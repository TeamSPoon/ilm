DATASET=roc_stories

#pushd data
#./get_${DATASET}.sh
#popd

for SPLIT in train valid
do
	python create_ilm_examples.py \
		${SPLIT} \
		data/sen_masks/${DATASET} \
		--seed 0 \
		--data_name ${DATASET} \
		--data_dir  data/${DATASET} \
		--data_split ${SPLIT} \
		--num_examples_per_document 2 \
		--mask_cls ilm.mask.mid_sentence.MaskMSentence
done