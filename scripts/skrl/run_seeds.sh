

SEEDS=(1618 3141 1729)
ENVS=(Quadcontrol-Direct-v0 Quadcontrol-meta-v0)
# TRAJ_TYPE=(hover straight circle lemniscate slalom)
# TRAJ_TYPE=(lemniscate)
TRAJ_TYPE=(straight) #circle)

experiment=v_max_test

# # hovering
# for SEED in "${SEEDS[@]}"
# do
#     echo "training with seed $SEED"
#     python train.py --seed $SEED --video --distributed --headless --max_iterations 500 --experiment test_v_max
# done


# mkdir -p $experiment


for traj in "${TRAJ_TYPE[@]}"
do    
    echo "training $traj"
    for SEED in "${SEEDS[@]}"
        do
            echo "training with seed $SEED"
            python train.py --task Quadcontrol-meta-v0 --seed $SEED --traj_type $traj \
            --max_iterations 300 --video --distributed --headless --experiment ${experiment}/${traj}
        done 
done



# for SEED in "${SEEDS[@]}"
# do
#     echo "training with seed $SEED"
#     python train.py --seed $SEED --task Quadcontrol-meta-v0 --traj_type straight --video --distributed --headless --max_iterations 1000 --experiment baseline_straight_randomised
# done

# for traj in "${TRAJ_TYPE[@]}"
# do    
#     echo "training $traj"
#     for SEED in "${SEEDS[@]}"
#         do
#             echo "training with seed $SEED"
#             python train.py --task Quadcontrol-meta-v0 --seed $SEED --traj_type $traj --max_iterations 1000 --video --distributed --headless --experiment baseline_a_$traj 
#         done
# done


# # individual learning for debugging
# traj=lemniscate

# for SEED in "${SEEDS[@]}"
# do
#     echo "training with seed $SEED"
#     python train.py --seed $SEED --task Quadcontrol-meta-v0 --traj_type $traj --video --distributed --headless --max_iterations 1000 --experiment baseline_gerono_$traj
# done

# Hydra commands for reference: we don't have to use seeds of other configs from parser, we can simply make use of hydra's ability to override config values from command line.
# with mulit-run:
# python train.py --multirun --task Quadcontrol-meta-v0 --traj_type straight, circle, lemniscate, slalom --seed 1618,3141,1729 --max_iterations 1000 --video --distributed --headless --experiment baseline_a
