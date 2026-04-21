

SEEDS=(1618 3141 1729)
ENVS=(Quadcontrol-Direct-v0 Quadcontrol-meta-v0)
TRAJ_TYPE=(straight circle lemniscate)
# TRAJ_TYPE=(circle)


# # hovering
# for SEED in "${SEEDS[@]}"
# do
#     echo "training with seed $SEED"
#     python train.py --seed $SEED --video --distributed --headless --max_iterations 500 --experiment baseline --device cuda:1
# done

if [ ! -d "baselines_tilt" ]; then
    mkdir -p baselines_tilt
fi

# baselines_tilt

for traj in "${TRAJ_TYPE[@]}"
do    
    echo "training $traj"
    for SEED in "${SEEDS[@]}"
        do
            echo "training with seed $SEED"
            python train.py --task Quadcontrol-Tilt-meta-v1 --seed $SEED --max_iterations 500 \
            --traj_type $traj --video --distributed --headless \
            --experiment baselines_tilt/${traj} \
            --device cuda:0
        done
done



# for SEED in "${SEEDS[@]}"
# do
#     echo "training with seed $SEED"
#     python train.py --seed $SEED --task Quadcontrol-NoGoal-Direct-v0 --video --distributed --headless --max_iterations 300 --experiment goal_nogoal 
# done