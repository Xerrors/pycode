#!/bin/bash

# # AdamW 优化器
# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# # log/SequeezeNetInTAU/1609144153_6974

# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=64 --num-worker=4 --epochs=100 --milestones="[100]"
# # log/SequeezeNetInTAU/1609145195_7080

# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# # log/SequeezeNetInTAU/1609146476_6826

# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# # log/SequeezeNetInTAU/1609147301_7142

python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"

python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[50]"

python -W ignore train_classify.py --base-lr=0.004 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"

python -W ignore train_classify.py --base-lr=0.004 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[50]"
# # None
python -W ignore train_classify.py --base-lr=0.01 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# # None
python -W ignore train_classify.py --base-lr=0.01 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[50]"
# # None
python -W ignore train_classify.py --base-lr=0.02 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# # None
python -W ignore train_classify.py --base-lr=0.02 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[50]"
# None

# SGD 优化器
# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.0001 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"


# python -W ignore train_classify.py --base-lr=0.01 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# python -W ignore train_classify.py --base-lr=0.004 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"
# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]"

# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[50]"

# python -W ignore train_classify.py --base-lr=0.001 --weight-decay=0.01 --batch-size=32 --num-worker=4 --epochs=100 --milestones="[100]" --nesterov
