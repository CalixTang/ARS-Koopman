#!/bin/bash
for i in {762...743}
do
   MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Hammer/Hammer_task.pickle --visualize False --save_fig False --config Samples/Hammer/CIMER/job_config.json --policy Samples/Hammer/CIMER/best_eval_sr_policy.pickle --koopman_path ./Calix_Testing/koopman_mats/hammer_mode_$i.npy --only_record_video
done

for i in {758...739}
do
   MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Relocate/Relocate_task.pickle --visualize False --save_fig False --config Samples/Relocate/CIMER/job_config.json --policy Samples/Relocate/CIMER/best_eval_sr_policy.pickle --koopman_path ./Calix_Testing/koopman_mats/relocate_mode_$i.npy --only_record_video
done

for i in {545...526}
do
   MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Door/Door_task.pickle --visualize False --save_fig False --config Samples/Door/CIMER/job_config.json --policy Samples/Door/CIMER/best_eval_sr_policy.pickle --koopman_path ./Calix_Testing/koopman_mats/door_mode_$i.npy --only_record_video
done