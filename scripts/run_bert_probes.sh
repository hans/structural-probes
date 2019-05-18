#!/bin/bash

#SBATCH -t 1-0
#SBATCH --mem 8G
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --qos=cpl
#SBATCH -a 1-8%2

# Run structural probes on a collection of BERT models at different timesteps.

SPROBE_DIR=~/om2/others/structural-probes
cd $SPROBE_DIR
shopt -s nullglob
PROBE_FILES=($SPROBE_DIR/en_ewt-ud/*.txt)

BERT_PROBE_DIR=$SPROBE_DIR/bert
YAML_TEMPLATE=$BERT_PROBE_DIR/bert.template.yaml

BERT_DIR=~/om2/others/bert
BASE_MODEL="uncased_L-12_H-768_A-12"
FINETUNE_DESC="finetune-250"
CKPTS=(5 100 200 250)
LAYERS=(0 6 11)
# FINETUNE_MODELS=(LM LM_scrambled QQP SQuAD)
# TARGET_RUNS=($(seq 1 8))
model=$TASK
run=${SLURM_ARRAY_TASK_ID}

BERT_CONTAINER=~/imgs/tf1.12-gpu.simg
SPROBE_CONTAINER=~/imgs/structural-probes.simg

# for model in ${FINETUNE_MODELS[*]}; do
#     for run in ${TARGET_RUNS[*]}; do
        for ckpt in ${CKPTS[*]}; do
            model_desc="${FINETUNE_DESC}.${BASE_MODEL}.${model}-run${run}"
            model_dir="$BERT_DIR/$model_desc"
            hdf5_desc="${model_desc}-${ckpt}"
            hdf5_dir="$BERT_PROBE_DIR/${hdf5_desc}"
            mkdir -p $hdf5_dir

            # Extract BERT model encodings
            pushd $BERT_DIR
            for probe_file in ${PROBE_FILES[*]}; do
                hdf5_name=`basename "$probe_file" | sed 's/txt/hdf5/'`
                echo "outputting to $hdf5_dir/$hdf5_name"
                singularity exec --nv -B /om -B /om2 $BERT_CONTAINER python extract_features.py \
                    --input_file=$probe_file --output_file="$hdf5_dir/$hdf5_name" \
                    --vocab_file="${model_dir}/vocab.txt" \
                    --bert_config_file="${model_dir}/bert_config.json" \
                    --init_checkpoint="${model_dir}/model.ckpt-${ckpt}" \
                    --layers="0,1,2,3,4,5,6,7,8,9,10,11" \
                    --output_format=hdf5 \
                    --max_seq_length 96 --batch_size 64
                echo $?
                echo "-------"
            done

            # Back to structural-probes.
            popd

            for layer in ${LAYERS[*]}; do
                outspec="$model_desc-$ckpt-layer$layer"
                outdir="$BERT_PROBE_DIR/$outspec"
                # Prepare YAML config file
                config_path=`echo $YAML_TEMPLATE | sed "s/template/$outspec/"`
                sed "s/<outdir>/$outspec/g" < $YAML_TEMPLATE | sed "s/<hdfdir>/$hdf5_desc/g" | sed "s/<layer>/$layer/g" > $config_path

                # Run structural probe training and evaluation
                singularity exec -B /om -B /om2 $SPROBE_CONTAINER python \
                    structural-probes/run_experiment.py --train-probe 1 $config_path || exit 1
            done

            # Remove the embedding files (they're big!)
            rm $outdir/*.hdf5
        done
    # done
# done
